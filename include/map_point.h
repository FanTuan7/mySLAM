
#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <memory>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <list>
#include <Eigen/Core>
#include <mutex>


class Mappoint
{
public:
//这句话干吗用的?
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    using Ptr = std::shared_ptr<Mappoint>;
    using ConstPtr = std::shared_ptr<const Mappoint>;

    //factory模式不会,用count来计数
    static unsigned long count;
    unsigned long _id;
    unsigned long _id_g2o;
    
    //cv::KeyPoint _kp;
    cv::Mat _descripter;
    //相对于所在帧的位姿
    std::unordered_map<int, Eigen::Vector3d> _T_c2p;
    Eigen::Vector3d _T_w2p;
    unsigned int observations;

    bool good;

    Mappoint(cv::Mat descripter);

    std::mutex _mutex;
    //对每一个frame都有一个kp
    std::unordered_map<int, cv::KeyPoint> _kps;
    //map.insert(std::make_pair(1, "Scala"));
};



#endif