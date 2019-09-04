#ifndef TRACKING_H
#define TRACKING_H

#include "camera.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include "frame.h"
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include "helper.h"
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "map_point.h"
#include "map.h"
#include "ORBmatcher.h"
#include "viewer.h"

class Tracking
{
    public:
    using Ptr = std::shared_ptr<Tracking>;
    using ConstPtr = std::shared_ptr<const Tracking>;

    enum TrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    TrackingState _state;
    Frame::Ptr _current_frame;
    Frame::Ptr _last_frame;

    Tracking(Map::Ptr map);

    void addFrame(Frame::Ptr frame);
    void track();
    //把第一帧存进_frist_frame,作为整个tracking的原点 之后改成地图初始化
    void stereoInitialization();
    
    //没有任何trick, 直接前后两帧 暴力匹配特征点,然后根据特征点的3D坐标,通过ICP求解相机运动.
    int trackLastFrameBF();
    //由于前后两帧的光照可能会不同, 所有前后两帧的使用都不用光流法.
    //通过之前两帧的相对运动,将上一帧的特征点投影到当前帧上,然后在投影范围内进行特征点匹配
    void trackWithMotionModel();

    //通过左右两个图像计算该帧的特征点和3D坐标
    void initCurrentFrame();
    //根据motion更新当前帧的世界坐标和3D点坐标
    void updateCurrentFrame();

    //未使用
    void trackLastFrameLK();
    void EstimateCurrentPose();

    //讲当前帧作为关键帧放到地图中
    void insertKeyframe();

    Sophus::SE3d _motion;
    bool _motion_is_set;

    Map::Ptr _map =nullptr;

    //至少100个特征点对匹配,才是OK
    const int TH_LOW = 100;

    int kfcounter;
    Frame::Ptr _KF;

    Viewer::Ptr _viewer;

};

#endif