#include "tracking.h"

Tracking::Tracking(Camera::ConstPtr camera,
                   Map::Ptr map,
                   int features,
                   float scaleFactor,
                   int levels,
                   int iniThFAST,
                   int minThFAST,
                   float KF_DoWrate,
                   float KF_mindistance,
                   const DBoW3::Vocabulary &vocab)
    : _camera(camera),
      _map(map),
      _features(features),
      _scaleFactor(scaleFactor),
      _levels(levels),
      _iniThFAST(iniThFAST),
      _minThFAST(minThFAST),
      _KF_DoWrate(KF_DoWrate),
      _KF_mindistance(KF_mindistance),
      _vocab(vocab)
{
    _state = NOT_INITIALIZED;
    _motion_is_set = false;
}

void Tracking::addFrame(Frame::Ptr frame)
{
    imshow("image", frame->_img_left);
    cv::waitKey(1);
    _current_frame = frame;
    initCurrentFrame();
    track();
    _last_frame = _current_frame;
}

void Tracking::track()
{
    if (_state == NOT_INITIALIZED)
    {
        stereoInitialization();
        _state = OK;
    }
    else if (_state == OK)
    {
        trackFramesBF(_last_frame, _current_frame);
    }
    else if (_state == LOST)
    {
    }
}

//将第一帧做双目匹配,得到特征点, 并且全都存放到地图中去
void Tracking::stereoInitialization()
{
    cout << "stereoInitialization" << endl;
    //第一帧的位姿为原点
    _current_frame->_T_c2w = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    _current_frame->_T_w2c = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    setKF();
}

//测试图片集,经过两个转弯,效果还可以. tracking 部分算是完成
//有BUG前两张图片的相对位姿 有时候会计算蹦掉
//BFmatching可以改进,比如提取两层特征点,
void Tracking::trackFramesBF(Frame::Ptr &f1, Frame::Ptr &f2)
{
    cout << endl
         << "trackLastFrameBF Frame " << f2->_id << " start. " << endl;

    ORBmatcher::Ptr matcher(new ORBmatcher());
    vector<DMatch> matches =
        matcher->BF_matching(f1->_img_left,
                             f2->_img_left,
                             f1->_kps_left,
                             f2->_kps_left,
                             f1->_ORB_descriptor_left,
                             f2->_ORB_descriptor_left);

    //ICP
    int size = matches.size();

    vector<Eigen::Vector3d> pt3f_1(size), pt3f_2(size);
    vector<Mappoint::Ptr> mps1 = f1->_map_points;
    vector<Mappoint::Ptr> mps2 = f2->_map_points;

    for (int i = 0; i < size; i++)
    {
        pt3f_1[i] = mps1[matches[i].queryIdx]->_localPos;
        pt3f_2[i] = mps2[matches[i].trainIdx]->_localPos;
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    //R和t是将相对于current的坐标系的位姿转换到last的坐标系中, 就是current相对于last的运动last_T_current
    pose_estimation_3d3d(pt3f_1, pt3f_2, R, t);

    //根据_motion更新当前帧和其中3D点的坐标

    _T_curr2last = Sophus::SE3d(R, t);
    _T_last2curr = _T_curr2last.inverse();
    _motion_is_set = true;

    updateCurrentFrame(matches);

    if (_KF_bestDoWScore < 0)
    {
        _KF_bestDoWScore = _vocab.score(_KF->_BoWv, _current_frame->_BoWv);
    }
    else
    {
        double score = _vocab.score(_KF->_BoWv, _current_frame->_BoWv);
        if (score < _KF_bestDoWScore * _KF_DoWrate)
        {
            c1 = true;
        }
    }

    if (c1)
    {
        setKF();
    }
}

void Tracking::setKF()
{
    _KF = _current_frame;
    _KF_bestDoWScore = -1;
    c1 = false;
    //c2 = false;
    insertKF();
    //_KFs.push_back(_current_frame);
}
//按照匀速模型将上一帧的特征点投影到当前帧,然后根据投影位置 范围查找ORB匹配
//最终得到3D匹配对,进行ICP计算
void Tracking::trackWithMotionModel()
{
    cout << endl
         << "trackWithMotionModel Frame start:" << _current_frame->_id << endl;
    // 1 利用匀速模型加速暴力匹配
    ORBmatcher::Ptr matcher(new ORBmatcher());
    vector<bool> rotation_check;
    vector<DMatch> matches =
        matcher->projection_Matching(_last_frame, _current_frame, _T_curr2last, rotation_check); //这个运动有问题

    if (matches.size() < 400)
    {
        _motion_is_set = false;
        _state = LOST;
        return;
    }

    //进过投影匹配的点还要进一步通过RANSAC的检测
    //2 用RANSAC优化匹配对
    vector<Point2f> pt2f1, pt2f2;

    for (int i = 0, size = matches.size(); i < size; i++)
    {
        pt2f1.push_back(_last_frame->_kps_left[matches[i].queryIdx].pt);
        pt2f2.push_back(_current_frame->_kps_left[matches[i].trainIdx].pt);
    }

    cv::Mat mask;
    findHomography(pt2f1, pt2f2, cv::RANSAC, 2, mask);

    vector<DMatch> good_matches;

    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    //if (size < TH_LOW)
    if (good_matches.size() < (matches.size() / 2))
    //if (good_matches.size() < 100)
    {
        _motion_is_set = false;
        _state = LOST;
        return;
    }

    vector<Eigen::Vector3d> pt3f_1(good_matches.size()), pt3f_2(good_matches.size());
    vector<Mappoint::Ptr> mps1 = _last_frame->_map_points;
    vector<Mappoint::Ptr> mps2 = _current_frame->_map_points;

    for (int i = 0, size = good_matches.size(); i < size; i++)
    {
        if (rotation_check[i])
        {
            pt3f_1[i] = mps1[matches[i].queryIdx]->_localPos;
            pt3f_2[i] = mps2[matches[i].trainIdx]->_localPos;
        }
        else
        {
            pt3f_1[i] = Eigen::Vector3d::Ones();
            pt3f_2[i] = Eigen::Vector3d::Ones();
        }
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    pose_estimation_3d3d(pt3f_2, pt3f_1, R, t);

    //匀速模型使用的速度
    _T_curr2last = Sophus::SE3d(R, t);

    //更新当前帧和3D点的坐标
    updateCurrentFrame(good_matches);

    //更新地图. 两部分地图点, 在上一帧共同观测的 和 这一帧单独的
    //根据matches将当前帧中所有匹配到的地图点的id换成前一阵对应地图点的id,然后插入所有地图点到地图中
    /*for (int i = 0,size = ; i < size; i++)
    {
        _current_frame->_map_points[matches[i].trainIdx]->_id = matches[i].queryIdx;
        //observations+1, 地图维护时根据观察数量决定删除哪些地图点
        _current_frame->_map_points[matches[i].trainIdx]->observations++;
    }
    int n = _current_frame->_map_points.size();
    for (int i = 0; i < n; i++)
    {
        _map->insertMapPoint(_current_frame->_map_points[i]);
    }
*/
}

void Tracking::initCurrentFrame()
{
    //这里的参数是用的ORB 的 Kitti的例子中的
    ORBextractor extractor_left(_features, _scaleFactor, _levels, _iniThFAST, _minThFAST);
    extractor_left(_current_frame->_img_left, _current_frame->_kps_left, _current_frame->_ORB_descriptor_left);

    ORBmatcher matcher;

    vector<KeyPoint> kps_right;
    vector<float> depths;

    //如果在右图匹配成功,那么depth的值为正,否则为-1
    depths = matcher.stereo_Matching(_current_frame->_img_left, _current_frame->_img_right,
                                     _current_frame->_kps_left, _camera->_fb);

    //删除没有成功匹配到的点
    int size = depths.size();

    vector<cv::KeyPoint> new_kps_left;
    cv::Mat new_ORB_descriptor_left;
    vector<float> new_depths;

    new_kps_left.reserve(size);
    new_depths.reserve(size);

    for (int i = 0; i < size; i++)
    {
        if (depths[i] > 0)
        {
            //直接删除,应该会降低速度
            new_kps_left.push_back(_current_frame->_kps_left[i]);
            new_ORB_descriptor_left.push_back(_current_frame->_ORB_descriptor_left.row(i));
            new_depths.push_back(depths[i]);
        }
    }

    _current_frame->_kps_left = new_kps_left;
    _current_frame->_ORB_descriptor_left = new_ORB_descriptor_left;
    _current_frame->AssignKeypointsToGrid();

    float fx = _camera->_fx;
    float fy = _camera->_fy;
    float cx = _camera->_cx;
    float cy = _camera->_cy;
    float x, y, z;

    int u, v;
    size = new_depths.size();
    for (int i = 0; i < size; i++)
    {
        z = new_depths[i];

        u = _current_frame->_kps_left[i].pt.x;
        v = _current_frame->_kps_left[i].pt.y;

        x = z * (u - cx) / fx;
        y = z * (v - cy) / fy;

        Mappoint::Ptr mp(new Mappoint(_current_frame->_ORB_descriptor_left.row(i), Eigen::Vector3d(x, y, z)));
        mp->_kps.insert(std::make_pair(_current_frame->_id, _current_frame->_kps_left[i]));
        _current_frame->_map_points.push_back(mp);
    }

    _vocab.transform(_current_frame->_ORB_descriptor_left, _current_frame->_BoWv);
}

//三个任务
//更新　当前帧世界坐标, 当前帧中地图点的3D坐标, 替换重复的地图点
void Tracking::updateCurrentFrame(vector<DMatch> &matches)
{

    _current_frame->_T_w2c = _last_frame->_T_w2c * _T_last2curr;
    _current_frame->_T_c2w = _T_curr2last * _last_frame->_T_c2w;

    for (auto mp : _current_frame->_map_points)
    {
        mp->_worldPos = _current_frame->_T_w2c * mp->_localPos;
    }

    int n = matches.size();
    for (int i = 0; i < n; i++)
    {
        Mappoint::Ptr mp1 = _last_frame->_map_points[matches[i].queryIdx];
        Mappoint::Ptr mp2 = _current_frame->_map_points[matches[i].trainIdx];

        mp1->_descripter = mp2->_descripter;
        //暂时不更新世界坐标,因为有可能之前观测的更准确. 反正之后localBA会调整.
        mp1->_kps.insert(std::make_pair(_current_frame->_id, _current_frame->_kps_left[i]));

        _current_frame->_map_points[matches[i].trainIdx] = mp1;
    }
    /*cout << endl
         << "当前帧的世界坐标 " << endl;
    cout << _current_frame->_T_w2c.rotationMatrix() << "\n"
         << endl;
    cout << _current_frame->_T_w2c.translation() << "\n"
         << endl;*/
}

//插入机制还有问题,应该基于以下点判断
//和上一个关键帧能有多少点匹配到, 按百分比来计算
Frame::Ptr Tracking::getKF()
{

    if (_KFs.empty())
    {
        return nullptr;
    }
    else
    {
        auto x = _KFs.front();
        _KFs.pop_front();
        return x;
    }
}

void Tracking::insertKF()
{
    _map->insertKeyFrame(_current_frame);
}
