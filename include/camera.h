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
    Camera(int id, double fx, double fy, double cx, double cy, double fb);
    
    int _id;
    double _fx, _fy, _cx, _cy, _fb;
    

    Eigen::Matrix3f K();
    Eigen::Matrix4f Tw2c();
    Eigen::Matrix4f Tc2w();
};

#endif