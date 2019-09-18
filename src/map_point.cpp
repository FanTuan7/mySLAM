#include "map_point.h"

//0 ~ 42 9496 7295
unsigned long Mappoint::count =0;

Mappoint::Mappoint(cv::Mat descripter)
:_descripter(descripter)
{   
    observations = 1;
    _id = count;
    count++;
}


