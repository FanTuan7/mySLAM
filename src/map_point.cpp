#include "map_point.h"

unsigned long Mappoint::count =0;

Mappoint::Mappoint(cv::Mat descripter)
:_descripter(descripter)
{   
    observations = 1;
    _id = count;
    count++;
    observations = 1;
}


