#include "local_mapping_g2o_type.h"

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;

LocalMapping_g2o::LocalMapping_g2o(Map::Ptr map)
    : _map(map)
{
    _KF_counter = 0;
}

void LocalMapping_g2o::run(Frame::Ptr frame)
{
    if (frame == nullptr)
    {
        return;
    }
    _currFrame = frame;
    //把当前帧和其中观察数大于1的地图点都放到待优化数据里
    _frames_BA.push_back(frame);
    for (auto mp : frame->_map_points)
    {
        //if (mp->observations > 1)
        {
            _mps_BA.insert(std::make_pair(mp->_id, mp));
        }
    }
    localBA();
    _frames_BA.clear();
    _mps_BA.clear();

    insert_KF_mps();
    /*if (frame->_isKF)
    {
        _KF_counter++;

        if (_KF_counter == 2)
        {
            _KF_counter = 1;
            localBA();
            
            //这应该可以写的更简洁,先凑活用
            _frames_BA.clear();
            _mps_BA.clear();
            _frames_BA.push_back(frame);
            for (auto mp : frame->_map_points)
            {
                if (mp->observations > 1)
                {
                    _mps_BA.insert(std::make_pair(mp->_id, mp));
                }
            }
        }
    _currFrame = frame;
    insert_KF_mps();
    }*/
}

void LocalMapping_g2o::insert_KF_mps()
{
    //永远只向地图插入当前帧, 因为frame都是指针操作,所以之前传送给地图的frame的位姿都被优化过了
    _map->insertKeyFrame(_currFrame);
    for (auto mp : _currFrame->_map_points)
    {
        if (mp->observations > 1)
        {
            _map->insertMapPoint(mp);
        }
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

//前半部分测试用 后半部分才是implementation
void LocalMapping_g2o::buildGraph(g2o::SparseOptimizer &o)
{

    ///////////////////////////////////////////////////////
    //////////one frame test///////////////////////////////
    g2o::VertexSE3Expmap *c = new g2o::VertexSE3Expmap();
    c->setId(0);
    //BUG
    //不应该给_T_w2c, 否则会导致1:把地图点固定之后优化结果完全飘走 2:不固定地图点, 优化结果总是在原点附近
    //没搞懂为什么应该给_T_c2w的坐标, 应该还是坐标变换这块的知识不扎实
    //c->setEstimate(g2o::SE3Quat(_currKF->_T_w2c.rotationMatrix(), _currKF->_T_w2c.translation()));
    c->setEstimate(g2o::SE3Quat(_currFrame->_T_c2w.rotationMatrix(), _currFrame->_T_c2w.translation()));
    o.addVertex(c);

    //添加地图点顶点,优化器中的坐标从1开始,一直到_map_points.size()+1
    for (int i = 0; i < _currFrame->_map_points.size(); i++)
    {
        g2o::VertexSBAPointXYZ *p = new g2o::VertexSBAPointXYZ();
        p->setId(i + 1);
        p->setEstimate(_currFrame->_map_points[i]->_T_w2p);
        if (_currFrame->_map_points[i]->observations > 1)
        {
            //p->setFixed(true);
        }
        p->setFixed(true); //全部点都固定,只优化相机位姿来减少重投影误差
        p->setMarginalized(true);
        o.addVertex(p);
    }

    g2o::CameraParameters *camera = new g2o::CameraParameters(718.856, Eigen::Vector2d(607.1928, 185.2157), 386.1448);
    camera->setId(0);
    o.addParameter(camera);

    for (int i = 0; i < _currFrame->_map_points.size(); i++)
    {
        auto mp = _currFrame->_map_points[i];

        g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
        e->setId(i);
        e->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(o.vertex(i + 1)));
        e->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(c));
        e->setMeasurement(Eigen::Vector2d(mp->_kps[_currFrame->_id].pt.x, mp->_kps[_currFrame->_id].pt.y));
        e->setInformation(Eigen::Matrix2d::Identity());
        //需要告诉g2o 相机内参
        e->setParameterId(0, 0);

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        e->setRobustKernel(rk);

        o.addEdge(e);
    }

    ////////////////////////////////////////////////////
    ////////////////local BA////////////////////////////

    /* int frames_size = _frames_BA.size();

    int index = 0;

    for (int i = 0; i < frames_size; ++i)
    {
        g2o::VertexSE3Expmap *c = new g2o::VertexSE3Expmap();
        c->setId(index++);
        c->setEstimate(g2o::SE3Quat(_frames_BA[i]->_T_c2w.rotationMatrix(), _frames_BA[i]->_T_c2w.translation()));
        o.addVertex(c);
    }

    //_mps_BA里面保存的都是至少被两帧观察到的地图点
    for (auto mp : _mps_BA)
    {
        g2o::VertexSBAPointXYZ *p = new g2o::VertexSBAPointXYZ();

        p->setId(index++);
        mp.second->_id_g2o = frames_size + index;
        p->setEstimate(mp.second->_T_w2p);
        p->setMarginalized(true);

        if (mp.second->observations > 1)
        {
            //p->setFixed(true);
        }
        p->setFixed(true); //全部点都固定,只优化相机位姿来减少重投影误差              BUG

        o.addVertex(p);
    }

    g2o::CameraParameters *camera = new g2o::CameraParameters(718.856, Eigen::Vector2d(607.1928, 185.2157), 386.1448);
    camera->setId(0);
    o.addParameter(camera);

    int edge_index = 0;
    for (int i = 0; i < frames_size; ++i)
    {
        auto f = _frames_BA[i];
        auto mps = f->_map_points;
        for (auto mp : mps)
        {
            if (mp->observations > 1)
            {
                g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
                e->setId(edge_index++);
                e->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(o.vertex(mp->_id_g2o)));
                e->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(o.vertex(i)));
                e->setMeasurement(Eigen::Vector2d(mp->_kps[f->_id].pt.x, mp->_kps[f->_id].pt.y));
                e->setInformation(Eigen::Matrix2d::Identity());
                e->setParameterId(0, 0);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                rk->setDelta(1.0);
                e->setRobustKernel(rk);

                o.addEdge(e);
            }
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
    ///////////////////////////////////////////////////////
    //////////one frame test///////////////////////////////
    g2o::VertexSE3Expmap *c = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    auto data = c->estimate();

    _currFrame->_T_c2w = Sophus::SE3d(data.rotation(), data.translation());
    _currFrame->_T_w2c = _currFrame->_T_c2w.inverse();

    cout << "当前帧优化后的世界坐标:" << endl;
    cout << _currFrame->_T_w2c.matrix3x4() << endl;

    ////////////////////////////////////////////////////
    ////////////////local BA////////////////////////////

    /*int frames_size = _frames_BA.size();
    int mps_size = _mps_BA.size();

    for (int i = 0; i < frames_size; i++)
    {
        cout << "Frame: " << _frames_BA[i]->_id << " 优化前的世界坐标" << endl;
        cout << _frames_BA[i]->_T_w2c.matrix3x4() << endl;

        g2o::VertexSE3Expmap *c = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
        auto data = c->estimate();
        _frames_BA[i]->_T_c2w = Sophus::SE3d(data.rotation(), data.translation());
        _frames_BA[i]->_T_w2c = _frames_BA[i]->_T_c2w.inverse();

        cout << "Frame: " << _frames_BA[i]->_id << " 优化后的世界坐标" << endl;
        cout << _frames_BA[i]->_T_w2c.matrix3x4() << endl;
    }
*/
    /* for (unordered_map<unsigned long, Mappoint::Ptr>::iterator iter = _mps_BA.begin(); iter != _mps_BA.end(); iter++)
    {

        g2o::VertexSBAPointXYZ *p = dynamic_cast<g2o::VertexSBAPointXYZ *>(_optimizer.vertex(iter->second->_id_g2o));
        iter->second->_worldPos = p->estimate();
    }
    */
}