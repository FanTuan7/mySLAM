#pragma once

#ifndef FRAME_H
#define FRAME_H

#include <camera.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include "helper.h"
#include "ORBextractor.h"
#include <math.h>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <map_point.h>
#include <list>
#include <mutex>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame
{
public:
    using Ptr = std::shared_ptr<Frame>;
    using ConstPtr = std::shared_ptr<const Frame>;
    bool _key_frame;

public:
    Camera::Ptr _camera;
    bool _isKF;
    unsigned long _id;
    double _time_stamp;
    Frame();
    Frame(long id, cv::Mat left, cv::Mat right);

    cv::Mat _img_left, _img_right;
    float _min_x;
    float _max_x;
    float _min_y;
    float _max_y;

    cv::Mat _ORB_descriptor_left;
    std::vector<cv::KeyPoint> _kps_left;

    void releaseImages();

    void test();
    //pose其实应该放在camera类里面
    Sophus::SE3d _pose;

    std::vector<Mappoint::Ptr> _map_points;

    std::vector<unsigned int>_Grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
    float _grid_height_inv;
    float _gird_width_inv;
    void AssignKeypointsToGrid();
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    vector<int> getFeaturesInArea(float &x, float  &y, float &r, int minLevel, int maxLevel);

    vector<float> _scaleFactors;

    std::mutex _mutex;
};

#endif
