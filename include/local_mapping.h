#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H


#include "frame.h"
#include "map.h"
#include "g2o_types.h"

#include <mutex>
#include <list>
#include <unordered_map>
#include <vector>
using namespace std;

class LocalMapping
{
    public:
    LocalMapping(Map::Ptr map);

    void readKF(Frame::Ptr KF);

    void insertKF();

    //当有两个关键帧就可以启动localBA
    

    Map::Ptr _map;

    Frame::Ptr _currKF=nullptr;
    Frame::Ptr _lastKF=nullptr;

    bool newFrame;
    bool readyForBA;

    void run();

    g2o::SparseOptimizer _optimizer;

    void initOptimizer();
    void buildGraph(std::vector<Frame::Ptr> &frames_BA,std::unordered_map<unsigned long, Mappoint::Ptr> &mps_BA);
    void localBA();
};


#endif