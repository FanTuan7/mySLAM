#ifndef VIEWER_H
#define VIEWER_H

#include <pangolin/pangolin.h>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "map.h"
#include "frame.h"
#include <thread>
#include <functional>
#include <unordered_map>
#include <atomic>
#include <list>
class Viewer
{
    public:

    const float BLUE[3] = {0, 0, 1};
    const float GREEN[3] = {0, 1, 0};
    const float RED[3] = {1, 0, 0};
    const float BLACK[3] = {0, 0, 0};

    using Ptr = std::shared_ptr<Viewer>;
    using ConstPtr = std::shared_ptr<const Viewer>;

    Viewer();

    void setMap(Map::Ptr map);

    void setCurrentFrame(Frame::Ptr frame);

    void updateLocalMap();

    void close();

    void threadLoop();

    void drawFrame(Frame::Ptr frame, const float* color);

    void drawMapPoints_keyframes();
    
    void followCurrentFrame(pangolin::OpenGlRenderState& vis_camera);

    Frame::Ptr _current_frame = nullptr;
    Map::Ptr _map = nullptr;

    std::thread _viewer_thread;

    std::atomic<bool> viewer_running;

    std::mutex _mutex;

    std::list<Frame::Ptr> _frames_to_draw;
    //std::unordered_map<unsigned long, Frame::Ptr> _frames_to_draw;
    std::list< Mappoint::Ptr> _mappoints_to_draw;

    cv::Mat plotFrameImage();

    std::vector<Eigen::Vector3d> _groundtruth;
    void drawGroundtruth();

};

#endif