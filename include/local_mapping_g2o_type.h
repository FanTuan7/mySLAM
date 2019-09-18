#ifndef LOCALMAPPINGG2OTYPE_H
#define LOCALMAPPINGG2OTYPE_H


#include "frame.h"
#include "map.h"

#include <mutex>
#include <list>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <typeinfo>

#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "g2o/core/robust_kernel_impl.h"
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;


//先用g2o的自带顶点和边试试
//目前只实现当前帧和上一帧的BA, 然后把优化后的frame和地图点再一起发送给map
class LocalMapping_g2o
{
    public:
    LocalMapping_g2o(Map::Ptr map);


    void readFrame(Frame::Ptr frame);

    //往地图里插入关键帧
    void insertFrame();

    //当有两个关键帧就可以启动localBA
    
    Map::Ptr _map;

    Frame::Ptr _currKF=nullptr;
    Frame::Ptr _lastKF=nullptr;

    bool newFrame;
    bool readyForBA;

    void run();

    //g2o::SparseOptimizer _optimizer;
    std::vector<Frame::Ptr> _frames_BA;
    std::unordered_map<unsigned long, Mappoint::Ptr> _mps_BA;

    void initOptimizer(g2o::SparseOptimizer &o);
    //根据_frames_BA和_mps_BA来构建优化图
    void buildGraph(g2o::SparseOptimizer &o);
    void localBA();


};


#endif