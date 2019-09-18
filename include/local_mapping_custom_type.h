#ifndef LOCALMAPPINGCUSTOMTYPE_H
#define LOCALMAPPINGCUSTOMTYPE_H

#include "frame.h"
#include "map.h"
#include "custom_types.h"

#include <mutex>
#include <list>
#include <unordered_map>
#include <vector>

using namespace std;

class LocalMapping
{
public:
    LocalMapping(Map::Ptr map);

    void readFrame(Frame::Ptr frame);

    void insertFrame();

    //当有两个关键帧就可以启动localBA

    Map::Ptr _map;

    Frame::Ptr _currKF = nullptr;
    Frame::Ptr _lastKF = nullptr;

    bool newFrame;
    bool readyForBA;

    void run();

    g2o::SparseOptimizer _optimizer;
    std::vector<Frame::Ptr> _frames_BA;
    std::unordered_map<unsigned long, Mappoint::Ptr> _mps_BA;

    void initOptimizer();
    void buildGraph();
    void localBA();
};

#endif