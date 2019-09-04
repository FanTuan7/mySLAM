
#ifndef MAP_H
#define MAP_H

#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include "map_point.h"
#include "frame.h"
#include <mutex>
#include <list>
class Map
{   
    public:
    using Ptr = std::shared_ptr<Map>;
    using ConstPtr = std::shared_ptr<const Map>;

    using MappointType = std::unordered_map<unsigned long, Mappoint::Ptr>;
    using FrameType = std::unordered_map<unsigned long, Frame::Ptr>;


    Map();

    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(Mappoint::Ptr map_point);

    std::mutex _mutex_frame;
    void set_current_frame(Frame::Ptr frame);
    Frame::Ptr _current_frame;

    std::mutex _mutex_mps;
    MappointType map_points_all;
    std::mutex _mutex_kfs;
    FrameType _keyframes_all;

    std::list< Mappoint::Ptr> _local_mappoints;
    std::list<Frame::Ptr> _local_keyframes;

    int _local_keyframes_num = 50; 
};

#endif