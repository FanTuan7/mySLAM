#include "local_mapping.h"

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolver;

LocalMapping::LocalMapping(Map::Ptr map)
    : _map(map)
{
}

void LocalMapping::readKF(Frame::Ptr KF)
{
    if (KF == nullptr)
    {
        newFrame = false;
        return;
    }

    newFrame = true;
    _lastKF = _currKF;
    _currKF = KF;

    if (_lastKF != nullptr)
    {
        //把所有frame串成一个链表
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
        localBA();
    }
    else if (newFrame)
    {
    }
}

void LocalMapping::insertKF()
{
    if (newFrame)
    {
        _map->insertKeyFrame(_currKF);

        vector<Mappoint::Ptr> mps = _currKF->_map_points;
        int size = mps.size();
        for (int j = 0; j < size; j++)
        {
            _map->insertMapPoint(mps[j]);
        }
    }
}

void LocalMapping::initOptimizer()
{

    g2o::LinearSolver<BlockSolver::PoseMatrixType> *linearSolver =
        new g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>();
    dynamic_cast<g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> *>(linearSolver)->setBlockOrdering(true);

    BlockSolver *solver_ptr = new BlockSolver(std::unique_ptr<BlockSolver::LinearSolverType>(linearSolver));

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<BlockSolver>(solver_ptr));

    _optimizer.setAlgorithm(solver);
}

void LocalMapping::buildGraph(std::vector<Frame::Ptr> &frames_BA, std::unordered_map<unsigned long, Mappoint::Ptr> &mps_BA)
{
    int frames_BA_size = frames_BA.size();
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
    }
}

void LocalMapping::localBA()
{

    //从当前关键帧开始，最多优化三帧
    std::vector<Frame::Ptr> frames_BA;
    std::unordered_map<unsigned long, Mappoint::Ptr> mps_BA;

    frames_BA.push_back(_currKF);
    frames_BA.push_back(_currKF->_lastKF);
    if (_currKF->_lastKF->_lastKF != nullptr)
    {
        frames_BA.push_back(_currKF->_lastKF->_lastKF);
    }

    unsigned long frames_size = frames_BA.size();
    for (int i = 0; i < frames_size; i++)
    {
        vector<Mappoint::Ptr> mps = frames_BA[i]->_map_points;
        int size = mps.size();
        for (int j = 0; j < size; j++)
        {
            if (mps_BA.find(mps[j]->_id) == mps_BA.end())
            {

                mps_BA.insert(make_pair(mps[j]->_id, mps[j]));
            }
        }
    }

    initOptimizer();
    buildGraph(frames_BA, mps_BA);

    _optimizer.initializeOptimization();
    _optimizer.setVerbose(true);
    _optimizer.optimize(10);

    //更新优化结果
    for (int i = 0; i < frames_size; i++)
    {
        VertexCamera *pCamera = dynamic_cast<VertexCamera *>(_optimizer.vertex(i));
        Eigen::Matrix<double, 6, 1> data = pCamera->estimate();

        Sophus::Vector6d se3;
        se3 << data[0], data[1], data[2], data[3], data[4], data[5];

        frames_BA[i]->_T_w2c = Sophus::SE3d::exp(se3);
        frames_BA[i]->_T_w2c = frames_BA[i]->_T_w2c.inverse();

        _map->insertKeyFrame(frames_BA[i]);
    }

    for (unordered_map<unsigned long, Mappoint::Ptr>::iterator iter = mps_BA.begin(); iter != mps_BA.end(); iter++)
    {

        Vertex3DPoint *pPoint = dynamic_cast<Vertex3DPoint *>(_optimizer.vertex(frames_size + iter->first));
        Eigen::Vector3d data = pPoint->estimate();

        iter->second->_worldPos = data;

        _map->insertMapPoint(iter->second);
    }
}