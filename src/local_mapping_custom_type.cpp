#include "local_mapping_custom_type.h"

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;

LocalMapping::LocalMapping(Map::Ptr map)
    : _map(map)
{
    _frames_BA.resize(2);
}

void LocalMapping::readFrame(Frame::Ptr frame)
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

void LocalMapping::run()
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

void LocalMapping::insertFrame()
{
    _map->insertKeyFrame(_currKF);

    int size = _currKF->_map_points.size();
    for (int j = 0; j < size; j++)
    {
        _map->insertMapPoint(_currKF->_map_points[j]);
    }
}

void LocalMapping::initOptimizer()
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    _optimizer.setAlgorithm(solver);
    _optimizer.setVerbose(true);
}

void LocalMapping::buildGraph()
{
    Eigen::Matrix<double, 6, 1> temVecCamera;
    Sophus::Vector6d se3 = _currKF->_T_w2c.log();
    for (int j = 0; j < 6; j++)
    {
        temVecCamera(j) = se3[j];
    }

    VertexCamera *pCamera = new VertexCamera();

    pCamera->setEstimate(temVecCamera);
    pCamera->setId(0);
    _optimizer.addVertex(pCamera);

    for (int i = 0, size = _currKF->_map_points.size(); i < size; i++)
    {
        Vertex3DPoint *p = new Vertex3DPoint();
        p->setId(i + 1);
        p->setEstimate(_currKF->_map_points[i]->_worldPos);
        p->setMarginalized(true);
        _optimizer.addVertex(p);
    }

    for (int i = 0; i < _currKF->_map_points.size(); ++i)
    {

        EdgeCamera_3Dpoint *e = new EdgeCamera_3Dpoint();

        e->setVertex(0, dynamic_cast<VertexCamera *>(_optimizer.vertex(0)));
        e->setVertex(1, dynamic_cast<Vertex3DPoint *>(_optimizer.vertex(i+1)));

        e->setInformation(Eigen::Matrix2d::Identity());

        float u = _currKF->_map_points[i]->_kps[_currKF->_id].pt.x;
        float v = _currKF->_map_points[i]->_kps[_currKF->_id].pt.y;
        e->setMeasurement(Eigen::Vector2d(u, v));

        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        e->setRobustKernel(rk);

        _optimizer.addEdge(e);
    }

    /*int frames_BA_size = frames_BA.size();
    int mps_BA_size = mps_BA.size();

    //设置位姿顶点
    for (int i = 0; i < frames_BA_size; ++i)
    {
        Eigen::Matrix<double, 6, 1> temVecCamera;
        Sophus::Vector6d se3 = frames_BA[i]->_T_w2c.log();
        for (int j = 0; j < 6; j++)
        {
            temVecCamera(j) = se3[j];
        }

        VertexCamera *pCamera = new VertexCamera();

        pCamera->setEstimate(temVecCamera);
        pCamera->setId(i);
        _optimizer.addVertex(pCamera);
    }

    //添加3D点的顶点
    for (unordered_map<unsigned long, Mappoint::Ptr>::iterator iter = mps_BA.begin(); iter != mps_BA.end(); iter++)
    {
        Vertex3DPoint *pPoint = new Vertex3DPoint();
        pPoint->setEstimate(iter->second->_worldPos);
        pPoint->setId(frames_BA_size + iter->first);
        pPoint->setMarginalized(true);
        _optimizer.addVertex(pPoint);
    }

    //添加边
    for (int i = 0; i < frames_BA_size; ++i)
    {

        EdgeCamera_3Dpoint *e = new EdgeCamera_3Dpoint();

        const int camera_id = i;

        auto mps = frames_BA[i]->_map_points;

        int n = mps.size();
        for (int j = 0; j < n; j++)
        {
            const unsigned long point_id = frames_BA_size + mps[j]->_id;

            e->setVertex(0, dynamic_cast<VertexCamera *>(_optimizer.vertex(camera_id)));
            e->setVertex(1, dynamic_cast<Vertex3DPoint *>(_optimizer.vertex(point_id)));

            e->setInformation(Eigen::Matrix2d::Identity());

            float u = mps[j]->_kps[frames_BA[i]->_id].pt.x;
            float v = mps[j]->_kps[frames_BA[i]->_id].pt.y;
            e->setMeasurement(Eigen::Vector2d(u, v));

            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);
            e->setRobustKernel(rk);

            _optimizer.addEdge(e);
        }
    }*/
}

void LocalMapping::localBA()
{

    initOptimizer();
    buildGraph();

    _optimizer.initializeOptimization();
    _optimizer.setVerbose(true);
    _optimizer.optimize(3);

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
    }*/
}