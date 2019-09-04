#include "tracking.h"

Tracking::Tracking(Map::Ptr map)
    : _map(map)
{
    _state = NOT_INITIALIZED;
    _motion_is_set = false;
    kfcounter = 0;
}

void Tracking::addFrame(Frame::Ptr frame)
{
    _current_frame = frame;
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
        if (_motion_is_set)
        {
            
            trackWithMotionModel();
 
            if (_state == LOST)
            {
                trackLastFrameBF();
                _state = OK;
            }
        }
        else
        {
            trackLastFrameBF();
        }
    }
    else if (_state == LOST)
    {
    }
    insertKeyframe();
    kfcounter++;
}

//将第一帧做双目匹配,得到特征点, 并且全都存放到地图中去
void Tracking::stereoInitialization()
{   cout << "stereoInitialization" <<endl;
    initCurrentFrame();
    //第一帧的位姿为原点
    _current_frame->_pose = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    
}

//测试图片集,经过两个转弯,效果还可以. tracking 部分算是完成
//有BUG前两张图片的相对位姿 有时候会计算蹦掉
//BFmatching可以改进,比如提取两层特征点,
int Tracking::trackLastFrameBF()
{   
    initCurrentFrame();

    //暴力匹配 + RANSAC 都在BF_matching 函数中
    ORBmatcher::Ptr matcher(new ORBmatcher());
    vector<DMatch> matches =
        matcher->BF_matching(_last_frame->_img_left,
                             _current_frame->_img_left,
                             _last_frame->_kps_left,
                             _current_frame->_kps_left,
                             _last_frame->_ORB_descriptor_left,
                             _current_frame->_ORB_descriptor_left);

    //ICP
    int size = matches.size();

    vector<Eigen::Vector3d> pt3f_1(size), pt3f_2(size);
    vector<Mappoint::Ptr> mps1 = _last_frame->_map_points;
    vector<Mappoint::Ptr> mps2 = _current_frame->_map_points;

    for (int i = 0; i < size; i++)
    {
        pt3f_1[i] = mps1[matches[i].queryIdx]->_localPos;
        pt3f_2[i] = mps2[matches[i].trainIdx]->_localPos;
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    pose_estimation_3d3d(pt3f_2, pt3f_1, R, t);

    //匀速模型使用的速度
    _motion = Sophus::SE3d(R, t);
    _motion_is_set = true;

    //更新当前帧和3D点的坐标
    updateCurrentFrame();

    //更新地图. 两部分地图点, 在上一帧共同观测的 和 这一帧单独的
    //根据matches将当前帧中所有匹配到的地图点的id换成前一阵对应地图点的id,然后插入所有地图点到地图中
    for (int i = 0; i < size; i++)
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

    //DEBUG
    cout << endl<< "trackLastFrameBF Frame:" << _current_frame->_id << endl;
    cout << "RANSAC匹配后的数量: " <<matches.size() << endl;
    cout << "相对速度" << endl << R << endl << t.transpose() << endl << endl; 

    return size;
}

//按照匀速模型将上一帧的特征点投影到当前帧,然后根据投影位置 范围查找ORB匹配
//最终得到3D匹配对,进行ICP计算
void Tracking::trackWithMotionModel()
{   
    initCurrentFrame();

    // 1 利用匀速模型加速暴力匹配
    ORBmatcher::Ptr matcher(new ORBmatcher());
    vector<bool> rotation_check;
    vector<DMatch> matches =
        matcher->projection_Matching(_last_frame, _current_frame, _motion, rotation_check);
    
    if (matches.size() < 400)
    {   
        _motion_is_set = false;
        _state = LOST;
        return;
    }

    //进过投影匹配的点还要进一步通过RANSAC的检测
    //2 用RANSAC优化匹配对
    vector<Point2f> pt2f1, pt2f2;

    for (int i = 0,size = matches.size(); i < size; i++)
    {
        pt2f1.push_back(_last_frame->_kps_left[matches[i].queryIdx].pt);
        pt2f2.push_back(_current_frame->_kps_left[matches[i].trainIdx].pt);
    }

    cv::Mat mask;
    findHomography(pt2f1, pt2f2, CV_RANSAC, 2, mask);

    vector<DMatch> good_matches;

    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    //if (size < TH_LOW)
    if (good_matches.size() < (matches.size()/2))
    //if (good_matches.size() < 100)
    {   
        _motion_is_set = false;
        _state = LOST;
        return;
    }

    vector<Eigen::Vector3d> pt3f_1(good_matches.size()), pt3f_2(good_matches.size());
    vector<Mappoint::Ptr> mps1 = _last_frame->_map_points;
    vector<Mappoint::Ptr> mps2 = _current_frame->_map_points;

    for (int i = 0,size=good_matches.size(); i < size; i++)
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
    _motion = Sophus::SE3d(R, t);
    _motion_is_set = true;

    cout << endl<< "trackWithMotionModel Frame:" << _current_frame->_id << endl;
    cout << "RANSAC匹配后的数量: " <<good_matches.size() << endl;
    cout << "相对速度" << endl << R << endl << t.transpose() << endl << endl; 
    //更新当前帧和3D点的坐标
    updateCurrentFrame();

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

void Tracking::trackLastFrameLK()
{
    initCurrentFrame();

    //1 根据光流寻找到上一帧的特征点在这一帧的投影
    Mat img1 = _last_frame->_img_left;
    Mat img2 = _current_frame->_img_left;
    vector<KeyPoint> keypoints1 = _last_frame->_kps_left;
    vector<KeyPoint> keypoints2 = _current_frame->_kps_left;

    //用光流法在右图追踪特征点,计算深度
    int size = keypoints1.size();
    vector<Point2f> points2f_1(size), points2f_2(size);

    for (int i = 0; i < size; i++)
    {
        points2f_1[i] = keypoints1[i].pt;
    }

    vector<uchar> status;
    Mat error;
    //看看两组点数量是否一致

    calcOpticalFlowPyrLK(img1, img2, points2f_1, points2f_2, status, error);

    vector<KeyPoint> kp1s, kp2s;
    for (size_t i = 0; i < status.size(); ++i)
    {
        //只保留在右图成功追踪到的点
        if (status[i])
        {

            KeyPoint kp1(points2f_1[i], 7);
            KeyPoint kp2(points2f_2[i], 7);
            kp1s.push_back(kp1);
            kp2s.push_back(kp2);
        }
    }

    // DEBUG>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    //查看右图光流追踪效果
    Mat out;

    for (int i = 0; i < keypoints1.size(); i++)
    {
        if (status[i])
        {
            if (abs(points2f_1[i].y - points2f_2[i].y) < 1)
            {
                circle(out, keypoints1[i].pt, 2, Scalar(0, 250, 0), 2);
                line(out, keypoints1[i].pt, keypoints2[i].pt, Scalar(0, 250, 0));
            }
        }
    }
    imshow("trackLastFrameLK", out);
    cvWaitKey(0);

    // 2
}

void Tracking::initCurrentFrame()
{
    //这里的参数是用的ORB 的 Kitti的例子中的
    ORBextractor extractor_left(2000, 1.2, 8, 20, 7);
    extractor_left(_current_frame->_img_left, _current_frame->_kps_left, _current_frame->_ORB_descriptor_left);
    _current_frame->_scaleFactors = extractor_left._vScaleFactor;

    ORBmatcher matcher;

    vector<KeyPoint> kps_right;
    std::vector<float> depth;
    //stereoMatching会删除左右图没有匹配到的特征点
    //所以会改变_kps_left,_ORB_descriptor_left的数据.
    matcher.stereo_Matching(_current_frame->_img_left, _current_frame->_img_right, _current_frame->_kps_left, kps_right, _current_frame->_camera->_fb, depth, _current_frame->_ORB_descriptor_left);

    _current_frame->AssignKeypointsToGrid();

    float fx = _current_frame->_camera->_fx;
    float fy = _current_frame->_camera->_fy;
    float cx = _current_frame->_camera->_cx;
    float cy = _current_frame->_camera->_cy;
    float x, y, z;
    int size = depth.size();
    int u, v;
    for (int i = 0; i < size; i++)
    {
        z = depth[i];

        u = _current_frame->_kps_left[i].pt.x;
        v = _current_frame->_kps_left[i].pt.y;

        x = z * (u - cx) / fx;
        y = z * (v - cy) / fy;

        Mappoint::Ptr mp(new Mappoint(_current_frame->_kps_left[i], _current_frame->_ORB_descriptor_left.row(i), Eigen::Vector3d(x, y, z)));
        _current_frame->_map_points.push_back(mp);
    }
}

//除了更新frame的世界坐标外,还要更新所有地图点的3D坐标
//计算当前帧的世界位姿
//更新地图点
void Tracking::updateCurrentFrame()
{
    if (_motion_is_set)
    {
        _current_frame->_pose = _motion * _last_frame->_pose;

        for (auto mp : _current_frame->_map_points)
        {
            mp->_worldPos = _current_frame->_pose * mp->_localPos;
        }
    }
}


void Tracking::insertKeyframe()
{   
    if(kfcounter%5 == 0 )
    {   
        cout << "************************************" << endl;
        cout << kfcounter%5 << "    " << kfcounter<<endl;
        _current_frame->_isKF = true;
        _map->insertKeyFrame(_current_frame);
        _viewer->updateLocalMap();
        _viewer->setCurrentFrame(_current_frame);
    }
}