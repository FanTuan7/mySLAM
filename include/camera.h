#ifndef CAMERA_H
#define CAMERA_H
//#include "myslam/common_include.h"
#include <memory> // shared_ptr
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

class Camera
{
public:
    using Ptr = std::shared_ptr<Camera>;
    using ConstPtr = std::shared_ptr<const Camera>;
    //Sophus::Vector6d _se3_pose;

    Camera();
    Camera(int id, float fx, float fy, float cx, float cy, float fb);
    
    int _id;
    float _fx, _fy, _cx, _cy, _fb;
    

    Eigen::Matrix3f K();
    Eigen::Matrix4f Tw2c();
    Eigen::Matrix4f Tc2w();
};

#endif