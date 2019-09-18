#include "local_mapping_g2o_type.h"

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;

LocalMapping_g2o::LocalMapping_g2o(Map::Ptr map)
    : _map(map)
{
    _frames_BA.resize(2);
}

void LocalMapping_g2o::readFrame(Frame::Ptr frame)
{
    if (frame == nullptr)
    {
        newFrame = false;
        return;
    }

    newFrame = true;
    _lastKF = _currKF;
    _currKF = frame;

    if (_lastKF != nullptr)
    {
        //把所有frame串成一个链表 目前还没有用到这个链表
        _lastKF->_nextKF = _currKF;
        _currKF->_lastKF = _lastKF;
        readyForBA = true;
    }
    else
    {
        readyForBA = false;
    }
}

void LocalMapping_g2o::run()
{
    if (readyForBA)
    {
        _frames_BA[0] = _lastKF;
        _frames_BA[1] = _currKF;

        _mps_BA.clear();
        for (int i = 0, size = _frames_BA.size(); i < size; i++)
        {
            auto mps = _frames_BA[i]->_map_points;
            //测试过, insert会忽略已经有key的数据
            for (auto mp : mps)
            {
                _mps_BA.insert(std::make_pair(mp->_id, mp));
            }
        }
        localBA();
        insertFrame();
    }
    else if (newFrame)
    {
    }
}

void LocalMapping_g2o::insertFrame()
{
    //永远只向地图插入当前帧, 因为frame都是指针操作,所以之前传送给地图的frame的位姿都被优化过了
    _map->insertKeyFrame(_currKF);

    int size = _currKF->_map_points.size();
    for (int j = 0; j < size; j++)
    {
        _map->insertMapPoint(_currKF->_map_points[j]);
    }
}

void LocalMapping_g2o::initOptimizer(g2o::SparseOptimizer &o)
{

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    //typedef g2o::LinearSolverCSparse<BlockSolverType::LandmarkMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    o.setAlgorithm(solver);
    o.setVerbose(true);
}

void LocalMapping_g2o::buildGraph(g2o::SparseOptimizer &o)
{

    //添加相机顶点, 优化器中的ID为0

    g2o::VertexSE3Expmap *c = new g2o::VertexSE3Expmap();
    c->setId(0);
    //c->setEstimate(g2o::SE3Quat(_currKF->_T_w2c.rotationMatrix(), _currKF->_T_w2c.translation()));
    c->setEstimate(g2o::SE3Quat(_currKF->_T_c2w.rotationMatrix(), _currKF->_T_c2w.translation()));
    //c->setEstimate(g2o::SE3Quat(Sophus::Matrix3d::Identity(), Sophus::Vector3d::Zero()));
    o.addVertex(c);

    //添加地图点顶点,优化器中的坐标从1开始,一直到_map_points.size()+1
    for (int i = 0; i < _currKF->_map_points.size(); i++)
    {
        g2o::VertexSBAPointXYZ *p = new g2o::VertexSBAPointXYZ();
        p->setId(i+1);
        p->setEstimate(_currKF->_map_points[i]->_T_w2p);
        if(_currKF->_map_points[i]->observations>1)
        {
           //p->setFixed(true);
        }
        p->setFixed(true); //全部点都固定,只优化相机位姿来减少重投影误差
        //p->setEstimate(mp->_T_c2p[_currKF->_id]);
        p->setMarginalized(true);
        o.addVertex(p);
    }

    //g2o::CameraParameters *camera = new g2o::CameraParameters(718.856, Eigen::Vector2d(607.1928, 185.2157), 386.1448);
    g2o::CameraParameters *camera = new g2o::CameraParameters(718.856, Eigen::Vector2d(607.1928, 185.2157), 386.1448);
    camera->setId(0);
    o.addParameter(camera);

    for (int i = 0; i < _currKF->_map_points.size(); i++)
    {
        auto mp = _currKF->_map_points[i]; //////

        g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
        e->setId(i);
        e->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(o.vertex(i+1))); ////////
        e->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(c));
        e->setMeasurement(Eigen::Vector2d(mp->_kps[_currKF->_id].pt.x, mp->_kps[_currKF->_id].pt.y)); ///////
        e->setInformation(Eigen::Matrix2d::Identity());
        //需要告诉g2o 相机内参
        e->setParameterId(0, 0);

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        e->setRobustKernel(rk);

        o.addEdge(e);
    }

    /*int frames_size = _frames_BA.size();

    int vertex_index = 0;
    //设置位姿顶点
    for (int i = 0; i < frames_size; ++i)
    {
        g2o::VertexSE3Expmap *c = new g2o::VertexSE3Expmap();
        c->setId(vertex_index);
        vertex_index++;
        c->setEstimate(g2o::SE3Quat(_frames_BA[i]->_T_w2c.rotationMatrix(), _frames_BA[i]->_T_w2c.translation()));

        _optimizer.addVertex(c);
    }

    //设置地图点顶点
    //不确定optimizer里面vertex的ID是否必须从0开始且连续?
    //懒得测试了
    //用笨办法, 在MapPoint类中再保存了它在g2o里面的ID
    //其实如果给frame保存g2o里面的顶点,应该运行更快

    for (auto mp : _mps_BA)
    {
        g2o::VertexSBAPointXYZ *p = new g2o::VertexSBAPointXYZ();

        p->setId(vertex_index);
        mp.second->_id_g2o = vertex_index;
        vertex_index++;
        p->setEstimate(mp.second->_worldPos);
        p->setMarginalized(true);

        _optimizer.addVertex(p);
    }

    g2o::CameraParameters *camera = new g2o::CameraParameters(718.856, Eigen::Vector2d(607.1928, 185.2157), 386.1448);
    camera->setId(0);
    _optimizer.addParameter(camera);

    int edge_index = 0;
    for (int i = 0; i < frames_size; ++i)
    {
        auto f = _frames_BA[i];
        auto mps = f->_map_points;
        for (auto mp : mps)
        {
            g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
            e->setId(edge_index);
            edge_index++;
            e->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(_optimizer.vertex(mp->_id_g2o)));
            e->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(_optimizer.vertex(i)));
            e->setMeasurement(Eigen::Vector2d(mp->_kps[f->_id].pt.x, mp->_kps[f->_id].pt.y));
            e->setInformation(Eigen::Matrix2d::Identity());
            //需要告诉g2o 相机内参
            e->setParameterId(0, 0);
            _optimizer.addEdge(e);
        }
    }*/
}

void LocalMapping_g2o::localBA()
{   
    g2o::SparseOptimizer optimizer;

    initOptimizer(optimizer);
    buildGraph(optimizer);

    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(10);

    //更新优化结果

    g2o::VertexSE3Expmap *c = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    auto data = c->estimate();

    _currKF->_T_c2w = Sophus::SE3d(data.rotation(), data.translation());
    _currKF-> _T_w2c= _currKF->_T_c2w.inverse();

    cout << "当前帧优化后的世界坐标:" << endl;
    cout << _currKF->_T_w2c.matrix3x4() << endl;

    /*for (int i = 0; i < _currKF->_map_points.size(); i++)
    {
        g2o::VertexSBAPointXYZ *p = dynamic_cast<g2o::VertexSBAPointXYZ *>(_optimizer.vertex(i + 1));
        Eigen::Vector3d _T_c2p = p->estimate();
        _currKF->_map_points[i]->_T_w2p = _T_c2p;
    }*/

    /*
    int frames_size = _frames_BA.size();
    int mps_size = _mps_BA.size();

    for (int i = 0; i < frames_size; i++)
    {
        g2o::VertexSE3Expmap *c = dynamic_cast<g2o::VertexSE3Expmap *>(_optimizer.vertex(i));
        auto data = c->estimate();
        cout << data << endl;
        _frames_BA[i]->_T_w2c = Sophus::SE3d(data.rotation(), data.translation());
        _frames_BA[i]->_T_c2w = _frames_BA[i]->_T_w2c.inverse();
    }

    for (unordered_map<unsigned long, Mappoint::Ptr>::iterator iter = _mps_BA.begin(); iter != _mps_BA.end(); iter++)
    {

        g2o::VertexSBAPointXYZ *p = dynamic_cast<g2o::VertexSBAPointXYZ *>(_optimizer.vertex(iter->second->_id_g2o));
        iter->second->_worldPos = p->estimate();
    }
    */
}