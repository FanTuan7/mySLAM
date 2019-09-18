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
        return;
    }

    if (_state == OK)
    {
        if (_motion_is_set == true)
        {
            //trackWithMotionModel();
            trackFramesBF(_last_frame, _current_frame);
        }
        else
        {
            trackFramesBF(_last_frame, _current_frame);
        }
    }
    if (_state == LOST)
    {
        trackFramesBF(_last_frame, _current_frame);
    }

    //insertKF();
}

//将第一帧做双目匹配,得到特征点, 并且全都存放到地图中去
void Tracking::stereoInitialization()
{
    cout << "stereoInitialization" << endl;
    //第一帧的位姿为原点
    _current_frame->_T_c2w = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    _current_frame->_T_w2c = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    for (auto mp : _current_frame->_map_points)
    {
        mp->_T_w2p =  mp->_T_c2p[_current_frame->_id];
        
    }

    //setKF();
}

//测试图片集,经过两个转弯,效果还可以. tracking 部分算是完成
//有BUG前两张图片的相对位姿 有时候会计算蹦掉
//BFmatching可以改进,比如提取两层特征点,
void Tracking::trackFramesBF(Frame::Ptr &f1, Frame::Ptr &f2)
{
    cout << endl
         << "trackLastFrameBF Frame " << f1->_id <<" and "<<f2->_id<< endl;

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
        pt3f_1[i] = mps1[matches[i].queryIdx]->_T_c2p[f1->_id];
        pt3f_2[i] = mps2[matches[i].trainIdx]->_T_c2p[f2->_id];
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    //R和t是将相对于current的坐标系的位姿转换到last的坐标系中, 就是current相对于last的运动last_T_current
    pose_estimation_3d3d(pt3f_2, pt3f_1, R, t);
    //pose_estimation_BA ( pt3f_1,pt3f_2, R, t);
    //根据_motion更新当前帧和其中3D点的坐标

    _T_last2curr = Sophus::SE3d(R, t);
    _T_curr2last = _T_last2curr.inverse();
    _motion_is_set = true;

    updateCurrentFrame(matches);

    /*
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
    }*/
    //Mat out;
    //drawMatches(f1->_img_left,f1->_kps_left,f2->_img_left,f2->_kps_left,matches,out);
    //imshow("last和current图匹配",out);
    //cv::waitKey(0);
    cout << endl<<"当前帧相对于上一帧的移动_" << endl;
    cout <<  _T_last2curr.matrix3x4() << endl<< endl;

    cout << endl<<"当前帧的世界坐标" << endl;
    cout <<  _current_frame->_T_w2c.matrix3x4() << endl<< endl;

}   

void Tracking::setKF()
{
    _KF = _current_frame;
    _KF_bestDoWScore = -1;
    c1 = false;
    //c2 = false;
    //_KFs.push_back(_current_frame);
}
//按照匀速模型将上一帧的特征点投影到当前帧,然后根据投影位置 范围查找ORB匹配
//最终得到3D匹配对,进行ICP计算
void Tracking::trackWithMotionModel()
{
    cout << endl
         << "trackWithMotionModel Frame : " << _current_frame->_id << endl;
    // 1 利用匀速模型加速暴力匹配
    ORBmatcher::Ptr matcher(new ORBmatcher());
    vector<bool> rotation_check;
    vector<DMatch> matches =
        matcher->projection_Matching(_last_frame, _current_frame, _T_curr2last, _T_last2curr, rotation_check, _camera); //这个运动有问题

    int after_prjMatch = matches.size();

    //进过投影匹配的点还要进一步通过RANSAC的检测
    //2 用RANSAC优化匹配对
    vector<Point2f> pt2f1, pt2f2;

    for (int i = 0, size = matches.size(); i < size; i++)
    {
        pt2f1.push_back(_last_frame->_kps_left[matches[i].queryIdx].pt);
        pt2f2.push_back(_current_frame->_kps_left[matches[i].trainIdx].pt);
    }

    cv::Mat mask;
    findHomography(pt2f1, pt2f2, cv::RANSAC, 4, mask);

    vector<DMatch> good_matches;

    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    int after_RANSAC = good_matches.size();

    if (after_RANSAC < after_prjMatch / 2)
    {
        _motion_is_set = false;
        _state = LOST;
        return;
    }

    vector<Eigen::Vector3d> pt3f_1(good_matches.size()), pt3f_2(good_matches.size());
    vector<Mappoint::Ptr> mps1 = _last_frame->_map_points;
    vector<Mappoint::Ptr> mps2 = _current_frame->_map_points;

    for(int i=0; i< good_matches.size() ; i++)
    {
        pt3f_1[i] = mps1[good_matches[i].queryIdx]->_T_c2p[_last_frame->_id];
        pt3f_2[i] = mps2[good_matches[i].trainIdx]->_T_c2p[_current_frame->_id];
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    //R和t是将相对于current的坐标系的位姿转换到last的坐标系中, 就是current相对于last的运动last_T_current
    pose_estimation_3d3d(pt3f_2, pt3f_1, R, t);
    //pose_estimation_BA ( pt3f_1,pt3f_2, R, t);
    //根据_motion更新当前帧和其中3D点的坐标

    _T_last2curr = Sophus::SE3d(R, t);
    _T_curr2last = _T_last2curr.inverse();
    _motion_is_set = true;

    updateCurrentFrame(matches);
}

void Tracking::initCurrentFrame()
{
    //这里的参数是用的ORB 的 Kitti的例子中的
    ORBextractor extractor_left(_features, _scaleFactor, _levels, _iniThFAST, _minThFAST);
    extractor_left(_current_frame->_img_left, _current_frame->_kps_left, _current_frame->_ORB_descriptor_left);

    ORBmatcher matcher;

    vector<KeyPoint> kps_right;
    vector<double> depths;

    //如果在右图匹配成功,那么depth的值为正,否则为-1
    depths = matcher.stereo_Matching(_current_frame->_img_left, _current_frame->_img_right,
                                     _current_frame->_kps_left, _camera->_fb);

    int size = depths.size();

    vector<cv::KeyPoint> new_kps_left;
    cv::Mat new_ORB_descriptor_left;
    vector<double> new_depths;

    new_kps_left.reserve(size);
    new_depths.reserve(size);

    for (int i = 0; i < size; i++)
    {
        if (depths[i] > 0)
        {
            new_kps_left.push_back(_current_frame->_kps_left[i]);
            new_ORB_descriptor_left.push_back(_current_frame->_ORB_descriptor_left.row(i));
            new_depths.push_back(depths[i]);
        }
    }

    _current_frame->_kps_left = new_kps_left;
    _current_frame->_ORB_descriptor_left = new_ORB_descriptor_left;
    _current_frame->AssignKeypointsToGrid();

    double fx = _camera->_fx;
    double fy = _camera->_fy;
    double cx = _camera->_cx;
    double cy = _camera->_cy;
    double x, y, z;

    int u, v;
    size = new_depths.size();
    for (int i = 0; i < size; i++)
    {
        z = new_depths[i];

        u = _current_frame->_kps_left[i].pt.x;
        v = _current_frame->_kps_left[i].pt.y;

        x = z * (u - cx) / fx;
        y = z * (v - cy) / fy;

        Mappoint::Ptr mp(new Mappoint(_current_frame->_ORB_descriptor_left.row(i)));
        
        mp->_T_c2p.insert(std::make_pair(_current_frame->_id, Eigen::Vector3d(x, y, z)));
        mp->_kps.insert(std::make_pair(_current_frame->_id, _current_frame->_kps_left[i]));
        _current_frame->_map_points.push_back(mp);
    }
    //_vocab.transform(_current_frame->_ORB_descriptor_left, _current_frame->_BoWv);
}

//三个任务
//更新　当前帧世界坐标, 当前帧中地图点的3D坐标, 替换重复的地图点
void Tracking::updateCurrentFrame(vector<DMatch> &matches)
{

    _current_frame->_T_w2c = _last_frame->_T_w2c * _T_last2curr;
    _current_frame->_T_c2w = _T_curr2last * _last_frame->_T_c2w; 

    for (auto mp : _current_frame->_map_points)
    {
        mp->_T_w2p = _current_frame->_T_w2c * mp->_T_c2p[_current_frame->_id];
        
    }

    //这段地图点的更新有问题
    //应该用上一帧的地图点替换掉当前帧的地图点
    int n = matches.size();
    for (int i = 0; i < n; i++)
    {
        Mappoint::Ptr mp1 = _last_frame->_map_points[matches[i].queryIdx];
        Mappoint::Ptr mp2 = _current_frame->_map_points[matches[i].trainIdx];

        mp1->_descripter = mp2->_descripter;
        //mp1->_T_c2p.insert(mp2->_T_c2p.begin(), mp2->_T_c2p.end()); //debug时候两句话的输出值不一样,但理论上应该没问题?
        mp1->_T_c2p.insert(std::make_pair(_current_frame->_id, mp2->_T_c2p[_current_frame->_id]));
        mp1->observations++;
        if(mp1->observations>2)
        {
            mp1->good = true;
        }
        mp1->_kps.insert(std::make_pair(_current_frame->_id, _current_frame->_kps_left[i]));

        _current_frame->_map_points[matches[i].trainIdx] = mp1;
    }
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
