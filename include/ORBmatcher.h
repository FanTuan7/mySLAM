#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include "map_point.h"
#include "frame.h"
#include "camera.h"
#include <cmath>

using namespace cv;
using namespace std;

class ORBmatcher
{
public:
    using Ptr = std::shared_ptr<ORBmatcher>;
    using ConstPtr = std::shared_ptr<const ORBmatcher>;

    ORBmatcher();

    void stereo_Matching(
        const Mat &img_left,
        const Mat &img_right,
        vector<KeyPoint> &keypoints_left,
        vector<KeyPoint> &keypoints_right,
        const float fb,
        vector<float> &depths,
        Mat &descriptors_left);

    //暂时用来做位姿预测
    vector<DMatch> BF_matching(const Mat &img1,
                              const Mat &img2,
                              vector<KeyPoint> &keypoints1,
                              vector<KeyPoint> &keypoints2,
                              Mat &descriptors1,
                              Mat &descriptors2);

    //按照匀速模型将上一帧的特征点投影到当前帧,然后根据投影位置 范围查找ORB匹配
    vector<DMatch> projection_Matching(Frame::Ptr frame1, Frame::Ptr frame2, Sophus::SE3d motion, vector<bool> &rotation_check);

    void ComputeThreeMaxima(const vector<vector<int>> &Hist, int &interval1, int &interval2, int &interval3);
    int descriptorDistance(const cv::Mat &a, const cv::Mat &b);

    int TH_HIGH = 100;
    int HISTO_LENGTH = 30;
};
#endif