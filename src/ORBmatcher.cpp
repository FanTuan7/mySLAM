#include "ORBmatcher.h"

ORBmatcher::ORBmatcher()
{
}

//测试完毕
//只在左右图匹配时使用
//用depth的值来表明该点是否在右图成功匹配到
vector<double> ORBmatcher::stereo_Matching(const Mat &img_left,
                                           const Mat &img_right,
                                           const vector<KeyPoint> &keypoints_left,
                                           const double &fb)
{
    int size = keypoints_left.size();

    //用光流法在右图追踪特征点,计算深度
    vector<double> depths(size);
    vector<Point2f> points2f_left(size), points2f_right(size);

    for (int i = 0; i < size; i++)
    {
        points2f_left[i] = keypoints_left[i].pt;
    }

    vector<uchar> status;
    Mat error;
    calcOpticalFlowPyrLK(img_left, img_right, points2f_left, points2f_right, status, error);

    size = status.size();
    for (int i = 0; i < size; i++)
    {
        //只保留在右图成功追踪到的点
        //status = 1 是成功寻找到的匹配
        if (status[i])
        {
            //保证满足基线约束 points2f的x和y为小数
            if (abs(points2f_left[i].y - points2f_right[i].y) < 1.5)
            {
                //控制视野范围,太近和太远的点都不考虑
                float disparity = points2f_left[i].x - points2f_right[i].x;
                double depth = fb / disparity;
                if (depth > 2 && depth < 45)
                {
                    depths[i] = depth;
                }
                else
                {
                    depths[i] = -1;
                }
            }
            else
            {
                //重复代码怎么解决?
                depths[i] = -1;
            }
        }
        else
        {
            depths[i] = -1;
        }
    }
    /*
    vector<KeyPoint> kp1s, kp2s;
    Mat out1, out2, compare;
     cvtColor(img_right, compare, cv::COLOR_GRAY2BGR);
    int m = depths.size();
    for(int i =0; i<m ;i++)
    {
        if(depths[i]>0)
        {
            kp1s.push_back(KeyPoint(points2f_left[i],5));
            kp2s.push_back(KeyPoint(points2f_right[i],5));
            line(compare, points2f_left[i], points2f_right[i], Scalar(0, 250, 0));
        }
    }

    drawKeypoints(img_left, kp1s, out1);
    drawKeypoints(img_right, kp2s, out2);
    imshow("left", out1);
    imshow("right", out2);
    imshow("compare", compare);
    cv::waitKey(0);
*/
    return depths;
}

vector<DMatch> ORBmatcher::BF_matching(const Mat &img1,
                                       const Mat &img2,
                                       const vector<KeyPoint> &keypoints1,
                                       const vector<KeyPoint> &keypoints2,
                                       const Mat &descriptors1,
                                       const Mat &descriptors2)
{
    //1 对两帧的特征点进行暴力匹配
    vector<DMatch> matches;
    BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors1, descriptors2, matches);

    //2 用RANSAC优化匹配对
    vector<Point2f> pt2f1, pt2f2;
    int size = matches.size();
    for (int i = 0; i < size; i++)
    {
        pt2f1.push_back(keypoints1[matches[i].queryIdx].pt);
        pt2f2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    cv::Mat mask;
    cv::findHomography(pt2f1, pt2f2, cv::RANSAC, 4, mask);

    vector<DMatch> good_matches;

    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    //DEBUG
    //cout << " good_matches: " << good_matches.size() << endl;
    /* Mat out1, out2;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, out1);
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, out2);

    //imshow("matches", out1);
    cv::imshow("good matches", out2);
    cv::waitKey(0);
*/
    return good_matches;
}

//根据投影加速匹配
//此时frame2的位姿已经按照motion调整过了
//假设相机运动符合匀速运动,那么frame1中的3D点经过motion变换,应该在frame2的相机坐标系中,
//然后通过内参得到3D点在frame2上的投影位置,然后寻找匹配
//如果两个点都匹配到一个点上怎么办? 应该能通过旋转直方图剔除出去
//把last的点按照

//   _current_frame->_T_w2c = _last_frame->_T_w2c * _T_last2curr;
//   _current_frame->_T_c2w = _T_curr2last * _last_frame->_T_c2w;

vector<DMatch> ORBmatcher::projection_Matching(Frame::Ptr frame1, Frame::Ptr frame2, Sophus::SE3d curr2last, Sophus::SE3d last2curr, vector<bool> &rotation_check, Camera::ConstPtr camera)
{
    //KITII车载相机一直是水平的, 所以不用统计旋转直方图, 直接判断两个点的旋转方向是否一致
    //bool checkOrientation = false;

    std::vector<Mappoint::Ptr> mps_1 = frame1->_map_points;
    int num_mps_1 = mps_1.size();

    //两个输出 results 和 rotation_check
    vector<DMatch> results;
    results.reserve(num_mps_1);
    rotation_check.clear();
    rotation_check.reserve(num_mps_1);

    double fx = camera->_fx;
    double fy = camera->_fy;
    double cx = camera->_cx;
    double cy = camera->_cy;

    //遍历frame1中的每一个mp
    for (int i = 0; i < num_mps_1; i++)
    {
        Mappoint::Ptr mp1 = mps_1[i];
        Eigen::Vector3d last2P = mp1->_T_c2p[frame1->_id];
        cout << last2curr.matrix3x4() << endl<<endl;
        cout << curr2last.matrix3x4() << endl<<endl;
        Eigen::Vector3d curr2P = curr2last.rotationMatrix() * last2P + curr2last.translation();////////////////////////TODO: 把这个坐标变换整明白

        double x = curr2P[0];
        double y = curr2P[1];
        double z_inv = 1.0 / curr2P[2];

        if (z_inv < 0)
        {
            continue;
        }

        float u = fx * x * z_inv + cx;
        float v = fy * y * z_inv + cy;

        if (u < frame2->_min_x || u > frame2->_max_x || v < frame2->_min_y || v > frame2->_max_y)
        {
            continue;
        }

        //根据图像所在层调整搜索半径
        int lastOctave = mp1->_kps[frame1->_id].octave;
        //15是ORB SLAM2中针对双目的值 应该是双目得到的3D点会不准确,所以范围大点
        //magic number 自己手动调的
        float radius = 15 /** frame2->_scaleFactors[lastOctave]*/;

        vector<int> index;

        //往前走物体会变大,所以用当前帧的小图像上的特征
        //金字塔模型是从1层开始, 图像不断缩小
        //有时车会停下来,这里懒得判断运动状态了, 直接-1 +1 搜索
        index = frame2->getFeaturesInArea(u, v, radius, lastOctave-1, lastOctave+1);

        if (index.empty())
        {
            continue;
        }

        //DEBUG 画图 看看左图单个特征点和右图的哪些特征点匹配
        Mat img1 = frame1->_img_left;
        Mat img2 = frame2->_img_left;

        vector<cv::KeyPoint> keypoints1, keypoints2, keypoints3;
        keypoints1.push_back(mp1->_kps[frame1->_id]); //左图kp
        keypoints2 = frame2->_kps_left; //右图对应的范围kp

        KeyPoint kp_right = KeyPoint(Point2f(u, v), 3);
        keypoints3.push_back(kp_right); //根据投影模型计算出来的右图kp

        cv::Mat out1, out2;

        vector<DMatch> matches;
        DMatch new_match;
        for (int m = 0; m < index.size(); m++)
        {
            DMatch new_match;
            new_match.queryIdx = 0;
            new_match.trainIdx = index[m];
            matches.push_back(new_match);
        }

        drawMatches(img1, keypoints1, img2, keypoints2, matches, out1);
        imshow("左图单个特征点与右图的区域匹配", out1);

        vector<DMatch> matches_0;
        DMatch match_0;
        match_0.queryIdx = 0;
        match_0.trainIdx = 0;
        matches_0.push_back(match_0);
        drawMatches(img1, keypoints1, img2, keypoints3, matches_0, out2);
        imshow("左图单个特征点与右图的单个匹配", out2);
        cv::waitKey(0);

        //开始暴力匹配
        cv::Mat d1 = mp1->_descripter;

        //利用secondBest来选取最明显的匹配
        int bestDist = 256;
        int secondBest = 256;
        int bestIndex = -1;

        for (int j = 0, max = index.size(); j < max; j++)
        {
            cv::Mat d2 = frame2->_map_points[index[j]]->_descripter;
            int dist = descriptorDistance(d1, d2);
            float angle_dist = abs( mp1->_kps[frame1->_id].angle - frame2->_map_points[index[j]]->_kps[frame2->_id].angle );

            if (dist < bestDist && angle_dist < 20)
            {   
                secondBest = bestDist;
                bestDist = dist;
                bestIndex = index[j];
            }
        }

        if ((secondBest-bestDist)>50)
        {
            DMatch match;
            match.queryIdx = i;
            match.trainIdx = bestIndex;
            match.distance = bestDist;
            results.push_back(match);
            rotation_check.push_back(true);

            /*if (checkOrientation)
            {
                float rot = mp1->_kps[frame1->_id].angle - frame2->_map_points[bestIndex2]->_kps[frame2->_id].angle;

                if (rot < 0)
                {
                    rot += 360.0;
                }

                int interval = round(rot * factor);
                if (interval == HISTO_LENGTH)
                {
                    interval = 0;
                }

                //用match在results的下标作为标识
                rotHist[interval].push_back(results.size() - 1);
            }*/
        }
    }

    /*if (checkOrientation)
    {

        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i != ind1 && i != ind2 && i != ind3)
            {
                for (int j = 0, lenght = rotHist[i].size(); j < lenght; j++)
                {
                    rotation_check[rotHist[i][j]] = false;
                }
            }
        }
    }*/

    //DEBUG
    //DEBUG 画图 看看左图的特征点能投影到右图的哪里

    /*   Mat img1 = frame1->_img_left;
        Mat img2 = frame2->_img_left;

        vector<cv::KeyPoint> keypoints1, keypoints2;
        keypoints1 = frame1->_kps_left;
        keypoints2 = frame2->_kps_left;

        cv::Mat out1, out2;
        cv::drawKeypoints(img1,keypoints1,out1);
        cv::drawKeypoints(img2,keypoints2,out2);*/
    //imshow("frame1的特征点",out1);
    //imshow("frame2的特征点",out2);

    /*  vector<DMatch> matches, good_matches;
    matches = results;
    for (int i = 0; i < rotation_check.size(); i++)
    {
        if (rotation_check[i])
        {
            good_matches.push_back(matches[i]);
        }
    }*/

    //cout << "      matches: " << matches.size() << endl;
    //cout << " good matches: " << good_matches.size() << endl;
    // cout << "投影匹配数量: " << good_matches.size() << endl;
    /*  cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, out1);
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, out2);

    //imshow("matches", out1);
    cv::imshow("good matches", out2);
    cv::waitKey(0);*/

    return results;
}

void ORBmatcher::ComputeThreeMaxima(const vector<vector<int>> &Hist, int &interval1, int &interval2, int &interval3)
{
    int max1; //最大
    int max2;
    int max3;

    for (int i = 0, length = Hist.size(); i < length; i++)
    {
        int x = Hist[i].size();

        if (x > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = x;
            interval3 = interval2;
            interval2 = interval1;
            interval1 = i;
        }
        else if (x > max2)
        {
            max3 = max2;
            max2 = x;
            interval3 = interval2;
            interval2 = i;
        }
        else if (x > max3)
        {
            max3 = x;
            interval3 = i;
        }
    }
}
// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
//每个描述子是由32个uchar型构成. 所以8行,

/*摘抄自上面的网站
The best method for counting bits in a 32-bit integer v is the following:

v = v - ((v >> 1) & 0x55555555);                    // reuse input as temporary
v = (v & 0x33333333) + ((v >> 2) & 0x33333333);     // temp
c = ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24; // count
*/
int ORBmatcher::descriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    //a 和 b 就是行向量,强制将a和b的每位变成32bit
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        //0x55555555 = 01010101 01010101 01010101 01010101
        v = v - ((v >> 1) & 0x55555555);
        //0x33333333 = 00110011 00110011 00110011 00110011
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}