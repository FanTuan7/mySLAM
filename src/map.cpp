#include "map.h"

Map::Map()
{

}


 void Map::insertKeyFrame(Frame::Ptr frame)
 {
     std::unique_lock<std::mutex> lck(_mutex_kfs);
    _keyframes_all.insert(std::make_pair(frame->_id,frame));
    _current_frame = frame;

    _local_keyframes.push_back(frame);
    if(_local_keyframes.size() > _local_keyframes_num)
    {
       // _local_keyframes.pop_front();
    }

    for(auto mp : frame->_map_points)
    {
        insertMapPoint(mp);
    }
 }


void Map::insertMapPoint(Mappoint::Ptr map_point)
{   

   std::unique_lock<std::mutex> lck(_mutex_mps);
    //插入新的关键点
    //if(map_points_all.find(map_point->_id) ==map_points_all.end()) //这个判断条件是多余的, 因为insert本身就会忽略ID重复的数据
    {
        map_points_all.insert(std::make_pair(map_point->_id,map_point));
    }
    //直接替换点旧的地图点
    //else
    { //写不写这句话都一样，因为local mapping里面是指针操作
    //    map_points_all[map_point->_id] = map_point;
    }
  
    _local_mappoints.push_back(map_point);
    if(_local_mappoints.size()>500000)
    {
        _local_mappoints.pop_front();
    }
    
    
}

void Map::set_current_frame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lck(_mutex_frame);
    _current_frame = frame;
}