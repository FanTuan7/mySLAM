#include "viewer.h"

using namespace std;

//从构造函数开始就画图
Viewer::Viewer()
{
    _viewer_thread = std::thread(std::bind(&Viewer::threadLoop, this));
}

void Viewer::close()
{
    viewer_running = false;
    _viewer_thread.join();
}

void Viewer::setMap(Map::Ptr map)
{
    _map = map;
}

void Viewer::setCurrentFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _current_frame = frame;
}

void Viewer::updateLocalMap()
{   
    assert(_map != nullptr);
    std::unique_lock<std::mutex> lock(_mutex);
    _frames_to_draw = _map->_local_keyframes;
    _mappoints_to_draw = _map->_local_mappoints;
}

void Viewer::threadLoop()
{
    pangolin::CreateWindowAndBind("MySLAM", 1024, 768);
    //以下属于opengl的指令, 先用着,之后再学
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));



    while (!pangolin::ShouldQuit() && viewer_running)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        std::unique_lock<std::mutex> lock(_mutex);
        if (_current_frame)
        {
            drawFrame(_current_frame, GREEN);
            followCurrentFrame(vis_camera);
            //画出当前帧的特征点和图像
            cv::Mat img = plotFrameImage();
            cv::imshow("image", img);
            cv::waitKey(1);
        }

        if (_map)
        {
            drawMapPoints_keyframes();
        }

        pangolin::FinishFrame();
        //us   办秒钟更新一下
        usleep(50000);
    }

    std::cout << "Stop viewer" << std::endl;
}

void Viewer::followCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    Sophus::SE3d Twc = _current_frame->_pose.inverse();
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);
}

void Viewer::drawFrame(Frame::Ptr frame, const float *color)
{
    //Sophus::SE3d Twc = frame->_pose.inverse();
    Sophus::SE3d Twc = frame->_pose;
    const float sz = 1.0;
    const int line_width = 2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;

    glPushMatrix();

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat *)m.data());

    if (color == nullptr)
    {
        glColor3f(1, 0, 0);
    }
    else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}

void Viewer::drawMapPoints_keyframes()
{
    for (auto& kf : _frames_to_draw) {
        drawFrame(kf, BLUE);
    }

    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto &mps : _mappoints_to_draw)
    {
        auto pos = mps->_worldPos;
        glColor3f(RED[0], RED[1], RED[2]);
        glVertex3d(pos[0], pos[1], pos[2]);
    }
    glEnd();
}

cv::Mat Viewer::plotFrameImage()
{
    cv::Mat img_out = _current_frame->_img_left;
    //cv::cvtColor(_current_frame->_img_left, img_out, CV_GRAY2BGR);

    for (size_t i = 0; i < _current_frame->_map_points.size(); ++i)
    {

        //少画几个点无所谓
        std::unique_lock<std::mutex> lock(_current_frame->_map_points[i]->_mutex, std::defer_lock);

        if (lock.try_lock())
        {
            auto kp = _current_frame->_map_points[i]->_kp;
            cv::circle(img_out, kp.pt, 2, cv::Scalar(0, 250, 0),
                       2);
        }
    }
    return img_out;
}