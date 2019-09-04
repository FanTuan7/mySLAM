#include "map.h"

Map::Map()
{

}


 void Map::insertKeyFrame(Frame::Ptr frame)
 {
     std::unique_lock<std::mutex> lck(_mutex_kfs);
    _keyframes_all.insert(std::make_pair(frame->_id,frame));
    _current_frame = frame;


    for(auto mp:frame->_map_points)
    {
        insertMapPoint(mp);
    }

    _local_keyframes.push_back(frame);
    if(_local_keyframes.size() > _local_keyframes_num)
    {
        _local_keyframes.pop_front();
    }
 }


void Map::insertMapPoint(Mappoint::Ptr map_point)
{   

   /* std::unique_lock<std::mutex> lck(_mutex_mps);
    //插入新的关键点
    if(map_points_all.find(map_point->_id) ==map_points_all.end())
    {
        map_points_all.insert(std::make_pair(map_point->_id,map_point));
    }
    //直接替换点旧的地图点
    else
    {
        map_points_all[map_point->_id] = map_point;
    }*/
    std::unique_lock<std::mutex> lck(_mutex_mps);
    _local_mappoints.push_back(map_point);
    if(_local_mappoints.size()>50000)
    {
        _local_mappoints.pop_front();
    }
    
    
}

void Map::set_current_frame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lck(_mutex_frame);
    _current_frame = frame;
}