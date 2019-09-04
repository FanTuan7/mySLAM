#include "map_point.h"

unsigned long Mappoint::count =0;

Mappoint::Mappoint(cv::KeyPoint kp,cv::Mat descripter,Eigen::Vector3d xyz)
:_kp(kp),_descripter(descripter),_localPos(xyz)
{   
    observations = 1;
    _id = count;
    count++;
}


