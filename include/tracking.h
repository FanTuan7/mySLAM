#ifndef TRACKING_H
#define TRACKING_H

#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <DBoW3/DBoW3.h>

#include "map_point.h"
#include "map.h"
#include "ORBmatcher.h"
#include "camera.h"
#include "frame.h"
#include "helper.h"

using namespace std;

class Tracking
{
public:
    using Ptr = std::shared_ptr<Tracking>;
    using ConstPtr = std::shared_ptr<const Tracking>;

    enum TrackingState
    {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    TrackingState _state;
    Frame::Ptr _current_frame;
    Frame::Ptr _last_frame;

    //tracking初始化时,要给它传递yaml中的参数,以及地图. 它只需要frame,ORB提取,匹配,地图,地图点,摄像头
    Tracking(Camera::ConstPtr camera,
             Map::Ptr map,
             int features,
             float scaleFactor,
             int levels,
             int iniThFAST,
             int minThFAST,
             float KF_DoWrate_Low,
             float KF_DoWrate_High,
             int KF_mindistance,
             int KF_maxdistance,
             const DBoW3::Vocabulary &vocab);

    void addFrame(Frame::Ptr frame);
    void track();
    //把第一帧存进_frist_frame,作为整个tracking的原点 之后改成地图初始化
    void stereoInitialization();

    //没有任何trick, 直接前后两帧 暴力匹配特征点,然后根据特征点的3D坐标,通过ICP求解相机运动.
    void trackFramesBF(Frame::Ptr &f1, Frame::Ptr &f2);
    //由于前后两帧的光照可能会不同, 所有前后两帧的使用都不用光流法.
    //通过之前两帧的相对运动,将上一帧的特征点投影到当前帧上,然后在投影范围内进行特征点匹配
    void trackWithMotionModel();

    //通过左右两个图像计算该帧的特征点和3D坐标
    void initCurrentFrame();
    //根据motion更新当前帧的世界坐标和3D点
    void updateCurrentFrame(vector<DMatch> &matches);

    Sophus::SE3d _T_curr2last;
    Sophus::SE3d _T_last2curr;
    bool _motion_is_set;

    //至少100个特征点对匹配,才是OK
    const int TH_LOW = 200;

    Camera::ConstPtr _camera;
    int _features;
    float _scaleFactor;
    int _levels;
    int _iniThFAST;
    int _minThFAST;
    Map::Ptr _map;

    void setKF();
    Frame::Ptr getKF();
    Frame::Ptr _KF;
    float _KF_DoWrate_Low;
    float _KF_DoWrate_High;
    int _KF_mindistance;
    int _KF_maxdistance;
    int _KF_distance;
    bool c1, c2, c3;
    list<Frame::Ptr> _KFs;

    DBoW3::Vocabulary _vocab;

    void insertKF();
};

#endif
