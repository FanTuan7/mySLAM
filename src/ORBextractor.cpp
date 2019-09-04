#include "ORBextractor.h"

//是为了防止计算描述子是越界
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 16;
//3这个余量是随便加的
const int EDGE_THRESHOLD = HALF_PATCH_SIZE + 3;
const double pi = 3.1415926;

using namespace std;
using namespace cv;

void computeAngle(const cv::Mat &image, vector<cv::KeyPoint> &keypoints)
{
    for (auto &kp : keypoints)
    {

        int half_patch_size = 8;
        int block_origin_x = kp.pt.x - half_patch_size;
        int block_origin_y = kp.pt.y - half_patch_size;
        cv::Mat block = image(cv::Rect(block_origin_x, block_origin_y, half_patch_size * 2, half_patch_size * 2));

        int m_01, m_10;
        for (int x = 0; x < block.cols; x++)
        {
            for (int y = 0; y < block.rows; y++)
            {
                m_01 += block.at<uchar>(y, x) * y;
                m_10 += block.at<uchar>(y, x) * x;
            }
        }
        //CV中的是弧度制
        kp.angle = atan2(m_10, m_01) * 180 / pi; // compute kp.angle
    }
    return;
}

int ORB_pattern[256 * 4] = {
    8, -3, 9, 5 /*mean (0), correlation (0)*/,
    4, 2, 7, -12 /*mean (1.12461e-05), correlation (0.0437584)*/,
    -11, 9, -8, 2 /*mean (3.37382e-05), correlation (0.0617409)*/,
    7, -12, 12, -13 /*mean (5.62303e-05), correlation (0.0636977)*/,
    2, -13, 2, 12 /*mean (0.000134953), correlation (0.085099)*/,
    1, -7, 1, 6 /*mean (0.000528565), correlation (0.0857175)*/,
    -2, -10, -2, -4 /*mean (0.0188821), correlation (0.0985774)*/,
    -13, -13, -11, -8 /*mean (0.0363135), correlation (0.0899616)*/,
    -13, -3, -12, -9 /*mean (0.121806), correlation (0.099849)*/,
    10, 4, 11, 9 /*mean (0.122065), correlation (0.093285)*/,
    -13, -8, -8, -9 /*mean (0.162787), correlation (0.0942748)*/,
    -11, 7, -9, 12 /*mean (0.21561), correlation (0.0974438)*/,
    7, 7, 12, 6 /*mean (0.160583), correlation (0.130064)*/,
    -4, -5, -3, 0 /*mean (0.228171), correlation (0.132998)*/,
    -13, 2, -12, -3 /*mean (0.00997526), correlation (0.145926)*/,
    -9, 0, -7, 5 /*mean (0.198234), correlation (0.143636)*/,
    12, -6, 12, -1 /*mean (0.0676226), correlation (0.16689)*/,
    -3, 6, -2, 12 /*mean (0.166847), correlation (0.171682)*/,
    -6, -13, -4, -8 /*mean (0.101215), correlation (0.179716)*/,
    11, -13, 12, -8 /*mean (0.200641), correlation (0.192279)*/,
    4, 7, 5, 1 /*mean (0.205106), correlation (0.186848)*/,
    5, -3, 10, -3 /*mean (0.234908), correlation (0.192319)*/,
    3, -7, 6, 12 /*mean (0.0709964), correlation (0.210872)*/,
    -8, -7, -6, -2 /*mean (0.0939834), correlation (0.212589)*/,
    -2, 11, -1, -10 /*mean (0.127778), correlation (0.20866)*/,
    -13, 12, -8, 10 /*mean (0.14783), correlation (0.206356)*/,
    -7, 3, -5, -3 /*mean (0.182141), correlation (0.198942)*/,
    -4, 2, -3, 7 /*mean (0.188237), correlation (0.21384)*/,
    -10, -12, -6, 11 /*mean (0.14865), correlation (0.23571)*/,
    5, -12, 6, -7 /*mean (0.222312), correlation (0.23324)*/,
    5, -6, 7, -1 /*mean (0.229082), correlation (0.23389)*/,
    1, 0, 4, -5 /*mean (0.241577), correlation (0.215286)*/,
    9, 11, 11, -13 /*mean (0.00338507), correlation (0.251373)*/,
    4, 7, 4, 12 /*mean (0.131005), correlation (0.257622)*/,
    2, -1, 4, 4 /*mean (0.152755), correlation (0.255205)*/,
    -4, -12, -2, 7 /*mean (0.182771), correlation (0.244867)*/,
    -8, -5, -7, -10 /*mean (0.186898), correlation (0.23901)*/,
    4, 11, 9, 12 /*mean (0.226226), correlation (0.258255)*/,
    0, -8, 1, -13 /*mean (0.0897886), correlation (0.274827)*/,
    -13, -2, -8, 2 /*mean (0.148774), correlation (0.28065)*/,
    -3, -2, -2, 3 /*mean (0.153048), correlation (0.283063)*/,
    -6, 9, -4, -9 /*mean (0.169523), correlation (0.278248)*/,
    8, 12, 10, 7 /*mean (0.225337), correlation (0.282851)*/,
    0, 9, 1, 3 /*mean (0.226687), correlation (0.278734)*/,
    7, -5, 11, -10 /*mean (0.00693882), correlation (0.305161)*/,
    -13, -6, -11, 0 /*mean (0.0227283), correlation (0.300181)*/,
    10, 7, 12, 1 /*mean (0.125517), correlation (0.31089)*/,
    -6, -3, -6, 12 /*mean (0.131748), correlation (0.312779)*/,
    10, -9, 12, -4 /*mean (0.144827), correlation (0.292797)*/,
    -13, 8, -8, -12 /*mean (0.149202), correlation (0.308918)*/,
    -13, 0, -8, -4 /*mean (0.160909), correlation (0.310013)*/,
    3, 3, 7, 8 /*mean (0.177755), correlation (0.309394)*/,
    5, 7, 10, -7 /*mean (0.212337), correlation (0.310315)*/,
    -1, 7, 1, -12 /*mean (0.214429), correlation (0.311933)*/,
    3, -10, 5, 6 /*mean (0.235807), correlation (0.313104)*/,
    2, -4, 3, -10 /*mean (0.00494827), correlation (0.344948)*/,
    -13, 0, -13, 5 /*mean (0.0549145), correlation (0.344675)*/,
    -13, -7, -12, 12 /*mean (0.103385), correlation (0.342715)*/,
    -13, 3, -11, 8 /*mean (0.134222), correlation (0.322922)*/,
    -7, 12, -4, 7 /*mean (0.153284), correlation (0.337061)*/,
    6, -10, 12, 8 /*mean (0.154881), correlation (0.329257)*/,
    -9, -1, -7, -6 /*mean (0.200967), correlation (0.33312)*/,
    -2, -5, 0, 12 /*mean (0.201518), correlation (0.340635)*/,
    -12, 5, -7, 5 /*mean (0.207805), correlation (0.335631)*/,
    3, -10, 8, -13 /*mean (0.224438), correlation (0.34504)*/,
    -7, -7, -4, 5 /*mean (0.239361), correlation (0.338053)*/,
    -3, -2, -1, -7 /*mean (0.240744), correlation (0.344322)*/,
    2, 9, 5, -11 /*mean (0.242949), correlation (0.34145)*/,
    -11, -13, -5, -13 /*mean (0.244028), correlation (0.336861)*/,
    -1, 6, 0, -1 /*mean (0.247571), correlation (0.343684)*/,
    5, -3, 5, 2 /*mean (0.000697256), correlation (0.357265)*/,
    -4, -13, -4, 12 /*mean (0.00213675), correlation (0.373827)*/,
    -9, -6, -9, 6 /*mean (0.0126856), correlation (0.373938)*/,
    -12, -10, -8, -4 /*mean (0.0152497), correlation (0.364237)*/,
    10, 2, 12, -3 /*mean (0.0299933), correlation (0.345292)*/,
    7, 12, 12, 12 /*mean (0.0307242), correlation (0.366299)*/,
    -7, -13, -6, 5 /*mean (0.0534975), correlation (0.368357)*/,
    -4, 9, -3, 4 /*mean (0.099865), correlation (0.372276)*/,
    7, -1, 12, 2 /*mean (0.117083), correlation (0.364529)*/,
    -7, 6, -5, 1 /*mean (0.126125), correlation (0.369606)*/,
    -13, 11, -12, 5 /*mean (0.130364), correlation (0.358502)*/,
    -3, 7, -2, -6 /*mean (0.131691), correlation (0.375531)*/,
    7, -8, 12, -7 /*mean (0.160166), correlation (0.379508)*/,
    -13, -7, -11, -12 /*mean (0.167848), correlation (0.353343)*/,
    1, -3, 12, 12 /*mean (0.183378), correlation (0.371916)*/,
    2, -6, 3, 0 /*mean (0.228711), correlation (0.371761)*/,
    -4, 3, -2, -13 /*mean (0.247211), correlation (0.364063)*/,
    -1, -13, 1, 9 /*mean (0.249325), correlation (0.378139)*/,
    7, 1, 8, -6 /*mean (0.000652272), correlation (0.411682)*/,
    1, -1, 3, 12 /*mean (0.00248538), correlation (0.392988)*/,
    9, 1, 12, 6 /*mean (0.0206815), correlation (0.386106)*/,
    -1, -9, -1, 3 /*mean (0.0364485), correlation (0.410752)*/,
    -13, -13, -10, 5 /*mean (0.0376068), correlation (0.398374)*/,
    7, 7, 10, 12 /*mean (0.0424202), correlation (0.405663)*/,
    12, -5, 12, 9 /*mean (0.0942645), correlation (0.410422)*/,
    6, 3, 7, 11 /*mean (0.1074), correlation (0.413224)*/,
    5, -13, 6, 10 /*mean (0.109256), correlation (0.408646)*/,
    2, -12, 2, 3 /*mean (0.131691), correlation (0.416076)*/,
    3, 8, 4, -6 /*mean (0.165081), correlation (0.417569)*/,
    2, 6, 12, -13 /*mean (0.171874), correlation (0.408471)*/,
    9, -12, 10, 3 /*mean (0.175146), correlation (0.41296)*/,
    -8, 4, -7, 9 /*mean (0.183682), correlation (0.402956)*/,
    -11, 12, -4, -6 /*mean (0.184672), correlation (0.416125)*/,
    1, 12, 2, -8 /*mean (0.191487), correlation (0.386696)*/,
    6, -9, 7, -4 /*mean (0.192668), correlation (0.394771)*/,
    2, 3, 3, -2 /*mean (0.200157), correlation (0.408303)*/,
    6, 3, 11, 0 /*mean (0.204588), correlation (0.411762)*/,
    3, -3, 8, -8 /*mean (0.205904), correlation (0.416294)*/,
    7, 8, 9, 3 /*mean (0.213237), correlation (0.409306)*/,
    -11, -5, -6, -4 /*mean (0.243444), correlation (0.395069)*/,
    -10, 11, -5, 10 /*mean (0.247672), correlation (0.413392)*/,
    -5, -8, -3, 12 /*mean (0.24774), correlation (0.411416)*/,
    -10, 5, -9, 0 /*mean (0.00213675), correlation (0.454003)*/,
    8, -1, 12, -6 /*mean (0.0293635), correlation (0.455368)*/,
    4, -6, 6, -11 /*mean (0.0404971), correlation (0.457393)*/,
    -10, 12, -8, 7 /*mean (0.0481107), correlation (0.448364)*/,
    4, -2, 6, 7 /*mean (0.050641), correlation (0.455019)*/,
    -2, 0, -2, 12 /*mean (0.0525978), correlation (0.44338)*/,
    -5, -8, -5, 2 /*mean (0.0629667), correlation (0.457096)*/,
    7, -6, 10, 12 /*mean (0.0653846), correlation (0.445623)*/,
    -9, -13, -8, -8 /*mean (0.0858749), correlation (0.449789)*/,
    -5, -13, -5, -2 /*mean (0.122402), correlation (0.450201)*/,
    8, -8, 9, -13 /*mean (0.125416), correlation (0.453224)*/,
    -9, -11, -9, 0 /*mean (0.130128), correlation (0.458724)*/,
    1, -8, 1, -2 /*mean (0.132467), correlation (0.440133)*/,
    7, -4, 9, 1 /*mean (0.132692), correlation (0.454)*/,
    -2, 1, -1, -4 /*mean (0.135695), correlation (0.455739)*/,
    11, -6, 12, -11 /*mean (0.142904), correlation (0.446114)*/,
    -12, -9, -6, 4 /*mean (0.146165), correlation (0.451473)*/,
    3, 7, 7, 12 /*mean (0.147627), correlation (0.456643)*/,
    5, 5, 10, 8 /*mean (0.152901), correlation (0.455036)*/,
    0, -4, 2, 8 /*mean (0.167083), correlation (0.459315)*/,
    -9, 12, -5, -13 /*mean (0.173234), correlation (0.454706)*/,
    0, 7, 2, 12 /*mean (0.18312), correlation (0.433855)*/,
    -1, 2, 1, 7 /*mean (0.185504), correlation (0.443838)*/,
    5, 11, 7, -9 /*mean (0.185706), correlation (0.451123)*/,
    3, 5, 6, -8 /*mean (0.188968), correlation (0.455808)*/,
    -13, -4, -8, 9 /*mean (0.191667), correlation (0.459128)*/,
    -5, 9, -3, -3 /*mean (0.193196), correlation (0.458364)*/,
    -4, -7, -3, -12 /*mean (0.196536), correlation (0.455782)*/,
    6, 5, 8, 0 /*mean (0.1972), correlation (0.450481)*/,
    -7, 6, -6, 12 /*mean (0.199438), correlation (0.458156)*/,
    -13, 6, -5, -2 /*mean (0.211224), correlation (0.449548)*/,
    1, -10, 3, 10 /*mean (0.211718), correlation (0.440606)*/,
    4, 1, 8, -4 /*mean (0.213034), correlation (0.443177)*/,
    -2, -2, 2, -13 /*mean (0.234334), correlation (0.455304)*/,
    2, -12, 12, 12 /*mean (0.235684), correlation (0.443436)*/,
    -2, -13, 0, -6 /*mean (0.237674), correlation (0.452525)*/,
    4, 1, 9, 3 /*mean (0.23962), correlation (0.444824)*/,
    -6, -10, -3, -5 /*mean (0.248459), correlation (0.439621)*/,
    -3, -13, -1, 1 /*mean (0.249505), correlation (0.456666)*/,
    7, 5, 12, -11 /*mean (0.00119208), correlation (0.495466)*/,
    4, -2, 5, -7 /*mean (0.00372245), correlation (0.484214)*/,
    -13, 9, -9, -5 /*mean (0.00741116), correlation (0.499854)*/,
    7, 1, 8, 6 /*mean (0.0208952), correlation (0.499773)*/,
    7, -8, 7, 6 /*mean (0.0220085), correlation (0.501609)*/,
    -7, -4, -7, 1 /*mean (0.0233806), correlation (0.496568)*/,
    -8, 11, -7, -8 /*mean (0.0236505), correlation (0.489719)*/,
    -13, 6, -12, -8 /*mean (0.0268781), correlation (0.503487)*/,
    2, 4, 3, 9 /*mean (0.0323324), correlation (0.501938)*/,
    10, -5, 12, 3 /*mean (0.0399235), correlation (0.494029)*/,
    -6, -5, -6, 7 /*mean (0.0420153), correlation (0.486579)*/,
    8, -3, 9, -8 /*mean (0.0548021), correlation (0.484237)*/,
    2, -12, 2, 8 /*mean (0.0616622), correlation (0.496642)*/,
    -11, -2, -10, 3 /*mean (0.0627755), correlation (0.498563)*/,
    -12, -13, -7, -9 /*mean (0.0829622), correlation (0.495491)*/,
    -11, 0, -10, -5 /*mean (0.0843342), correlation (0.487146)*/,
    5, -3, 11, 8 /*mean (0.0929937), correlation (0.502315)*/,
    -2, -13, -1, 12 /*mean (0.113327), correlation (0.48941)*/,
    -1, -8, 0, 9 /*mean (0.132119), correlation (0.467268)*/,
    -13, -11, -12, -5 /*mean (0.136269), correlation (0.498771)*/,
    -10, -2, -10, 11 /*mean (0.142173), correlation (0.498714)*/,
    -3, 9, -2, -13 /*mean (0.144141), correlation (0.491973)*/,
    2, -3, 3, 2 /*mean (0.14892), correlation (0.500782)*/,
    -9, -13, -4, 0 /*mean (0.150371), correlation (0.498211)*/,
    -4, 6, -3, -10 /*mean (0.152159), correlation (0.495547)*/,
    -4, 12, -2, -7 /*mean (0.156152), correlation (0.496925)*/,
    -6, -11, -4, 9 /*mean (0.15749), correlation (0.499222)*/,
    6, -3, 6, 11 /*mean (0.159211), correlation (0.503821)*/,
    -13, 11, -5, 5 /*mean (0.162427), correlation (0.501907)*/,
    11, 11, 12, 6 /*mean (0.16652), correlation (0.497632)*/,
    7, -5, 12, -2 /*mean (0.169141), correlation (0.484474)*/,
    -1, 12, 0, 7 /*mean (0.169456), correlation (0.495339)*/,
    -4, -8, -3, -2 /*mean (0.171457), correlation (0.487251)*/,
    -7, 1, -6, 7 /*mean (0.175), correlation (0.500024)*/,
    -13, -12, -8, -13 /*mean (0.175866), correlation (0.497523)*/,
    -7, -2, -6, -8 /*mean (0.178273), correlation (0.501854)*/,
    -8, 5, -6, -9 /*mean (0.181107), correlation (0.494888)*/,
    -5, -1, -4, 5 /*mean (0.190227), correlation (0.482557)*/,
    -13, 7, -8, 10 /*mean (0.196739), correlation (0.496503)*/,
    1, 5, 5, -13 /*mean (0.19973), correlation (0.499759)*/,
    1, 0, 10, -13 /*mean (0.204465), correlation (0.49873)*/,
    9, 12, 10, -1 /*mean (0.209334), correlation (0.49063)*/,
    5, -8, 10, -9 /*mean (0.211134), correlation (0.503011)*/,
    -1, 11, 1, -13 /*mean (0.212), correlation (0.499414)*/,
    -9, -3, -6, 2 /*mean (0.212168), correlation (0.480739)*/,
    -1, -10, 1, 12 /*mean (0.212731), correlation (0.502523)*/,
    -13, 1, -8, -10 /*mean (0.21327), correlation (0.489786)*/,
    8, -11, 10, -6 /*mean (0.214159), correlation (0.488246)*/,
    2, -13, 3, -6 /*mean (0.216993), correlation (0.50287)*/,
    7, -13, 12, -9 /*mean (0.223639), correlation (0.470502)*/,
    -10, -10, -5, -7 /*mean (0.224089), correlation (0.500852)*/,
    -10, -8, -8, -13 /*mean (0.228666), correlation (0.502629)*/,
    4, -6, 8, 5 /*mean (0.22906), correlation (0.498305)*/,
    3, 12, 8, -13 /*mean (0.233378), correlation (0.503825)*/,
    -4, 2, -3, -3 /*mean (0.234323), correlation (0.476692)*/,
    5, -13, 10, -12 /*mean (0.236392), correlation (0.475462)*/,
    4, -13, 5, -1 /*mean (0.236842), correlation (0.504132)*/,
    -9, 9, -4, 3 /*mean (0.236977), correlation (0.497739)*/,
    0, 3, 3, -9 /*mean (0.24314), correlation (0.499398)*/,
    -12, 1, -6, 1 /*mean (0.243297), correlation (0.489447)*/,
    3, 2, 4, -8 /*mean (0.00155196), correlation (0.553496)*/,
    -10, -10, -10, 9 /*mean (0.00239541), correlation (0.54297)*/,
    8, -13, 12, 12 /*mean (0.0034413), correlation (0.544361)*/,
    -8, -12, -6, -5 /*mean (0.003565), correlation (0.551225)*/,
    2, 2, 3, 7 /*mean (0.00835583), correlation (0.55285)*/,
    10, 6, 11, -8 /*mean (0.00885065), correlation (0.540913)*/,
    6, 8, 8, -12 /*mean (0.0101552), correlation (0.551085)*/,
    -7, 10, -6, 5 /*mean (0.0102227), correlation (0.533635)*/,
    -3, -9, -3, 9 /*mean (0.0110211), correlation (0.543121)*/,
    -1, -13, -1, 5 /*mean (0.0113473), correlation (0.550173)*/,
    -3, -7, -3, 4 /*mean (0.0140913), correlation (0.554774)*/,
    -8, -2, -8, 3 /*mean (0.017049), correlation (0.55461)*/,
    4, 2, 12, 12 /*mean (0.01778), correlation (0.546921)*/,
    2, -5, 3, 11 /*mean (0.0224022), correlation (0.549667)*/,
    6, -9, 11, -13 /*mean (0.029161), correlation (0.546295)*/,
    3, -1, 7, 12 /*mean (0.0303081), correlation (0.548599)*/,
    11, -1, 12, 4 /*mean (0.0355151), correlation (0.523943)*/,
    -3, 0, -3, 6 /*mean (0.0417904), correlation (0.543395)*/,
    4, -11, 4, 12 /*mean (0.0487292), correlation (0.542818)*/,
    2, -4, 2, 1 /*mean (0.0575124), correlation (0.554888)*/,
    -10, -6, -8, 1 /*mean (0.0594242), correlation (0.544026)*/,
    -13, 7, -11, 1 /*mean (0.0597391), correlation (0.550524)*/,
    -13, 12, -11, -13 /*mean (0.0608974), correlation (0.55383)*/,
    6, 0, 11, -13 /*mean (0.065126), correlation (0.552006)*/,
    0, -1, 1, 4 /*mean (0.074224), correlation (0.546372)*/,
    -13, 3, -9, -2 /*mean (0.0808592), correlation (0.554875)*/,
    -9, 8, -6, -3 /*mean (0.0883378), correlation (0.551178)*/,
    -13, -6, -8, -2 /*mean (0.0901035), correlation (0.548446)*/,
    5, -9, 8, 10 /*mean (0.0949843), correlation (0.554694)*/,
    2, 7, 3, -9 /*mean (0.0994152), correlation (0.550979)*/,
    -1, -6, -1, -1 /*mean (0.10045), correlation (0.552714)*/,
    9, 5, 11, -2 /*mean (0.100686), correlation (0.552594)*/,
    11, -3, 12, -8 /*mean (0.101091), correlation (0.532394)*/,
    3, 0, 3, 5 /*mean (0.101147), correlation (0.525576)*/,
    -1, 4, 0, 10 /*mean (0.105263), correlation (0.531498)*/,
    3, -6, 4, 5 /*mean (0.110785), correlation (0.540491)*/,
    -13, 0, -10, 5 /*mean (0.112798), correlation (0.536582)*/,
    5, 8, 12, 11 /*mean (0.114181), correlation (0.555793)*/,
    8, 9, 9, -6 /*mean (0.117431), correlation (0.553763)*/,
    7, -4, 8, -12 /*mean (0.118522), correlation (0.553452)*/,
    -10, 4, -10, 9 /*mean (0.12094), correlation (0.554785)*/,
    7, 3, 12, 4 /*mean (0.122582), correlation (0.555825)*/,
    9, -7, 10, -2 /*mean (0.124978), correlation (0.549846)*/,
    7, 0, 12, -2 /*mean (0.127002), correlation (0.537452)*/,
    -1, -6, 0, -11 /*mean (0.127148), correlation (0.547401)*/
};

void computeORBDesc(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    // 256/8bit = 32
    //每行是一个特征点的描述子,一个uchar可以表示8 bit,然后每行有32个uchar
    descriptors.create((int)keypoints.size(),32,CV_8UC1);
    //descriptors.resize(keypoints.size());
    for (int j = 0; j < keypoints.size(); j++)
    {
        vector<bool> d(256);
        cv::KeyPoint kp = keypoints[j];
        for (int i = 0; i < 256; i++)
        {
            int u_p = ORB_pattern[i * 4];
            int v_p = ORB_pattern[i * 4 + 1];
            int u_q = ORB_pattern[i * 4 + 2];
            int v_q = ORB_pattern[i * 4 + 3];

            int _u_p = (int)(cos(kp.angle / 180 * pi) * u_p - sin(kp.angle / 180 * pi) * v_p);
            int _v_p = (int)(sin(kp.angle / 180 * pi) * u_p + cos(kp.angle / 180 * pi) * v_p);
            int _u_q = (int)(cos(kp.angle / 180 * pi) * u_q - sin(kp.angle / 180 * pi) * v_q);
            int _v_q = (int)(sin(kp.angle / 180 * pi) * u_q + cos(kp.angle / 180 * pi) * v_q);

            uchar p = image.at<uchar>(kp.pt.y + _v_p, kp.pt.x + _u_p);
            uchar q = image.at<uchar>(kp.pt.y + _v_q, kp.pt.x + _u_q);
            d[i] = (p > q);
        }
        uchar* desc =descriptors.ptr((int)j);
        for (int i = 0; i < 32; i++)
        {
            int x;
            x = d[i*8 + 0];
            x |= d[i*8 + 1] <<1;
            x |= d[i*8 + 2] <<2;
            x |= d[i*8 + 3] <<3;
            x |= d[i*8 + 4] <<4;
            x |= d[i*8 + 5] <<5;
            x |= d[i*8 + 6] <<6;
            x |= d[i*8 + 7] <<7;
            desc[i] = (uchar) x;
        }

        
    }
    return;
}



ORBextractor::ORBextractor(int nfeatures, float scaleFactor, int nlevels,int iniThFAST, int minThFAST)
:
_nfeatures(nfeatures), _scaleFactor(scaleFactor), _nlevels(nlevels),
_iniThFAST(iniThFAST), _minThFAST(minThFAST)
{ //预先留存位置.加快计算时间
    _vScaleFactor.resize(_nlevels);
    _vLevelSigma2.resize(_nlevels);
    //0层是原始图像
    _vScaleFactor[0] = 1.0f;
    _vLevelSigma2[0] = 1.0f;
    //确定每一层和原始图像的比例,方便后续计算
    for (int i = 1; i < _nlevels; i++)
    {
        _vScaleFactor[i] = _vScaleFactor[i - 1] * scaleFactor;
        _vLevelSigma2[i] = _vScaleFactor[i] * _vScaleFactor[i];
    }

    _vInvScaleFactor.resize(nlevels);
    _vInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++)
    {
        _vInvScaleFactor[i] = 1.0f / _vScaleFactor[i];
        _vInvLevelSigma2[i] = 1.0f / _vLevelSigma2[i];
    }

    _vImagePyramid.resize(_nlevels);

    _nFeaturesPerLevel.resize(_nlevels);
    float factor = 1.0f / _scaleFactor;
    //(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels)) 等比数列求和公式
    //这行是金字塔0层的特征数
    float nDesiredFeatubeiPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    //前几层的feature数都是按比例算的,最后一层是看还剩下多少特征点数.
    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++)
    {
        _nFeaturesPerLevel[level] = cvRound(nDesiredFeatubeiPerScale);
        sumFeatures += _nFeaturesPerLevel[level];
        nDesiredFeatubeiPerScale *= factor;
    }
    //剩下还有多少的特征点都放在最后一层,确保不小于0
    _nFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
}

//重载()操作符,完成对图像的金字塔分层,特征点计算,特征点分布, 描述子计算
void ORBextractor::operator()(InputArray imageRaw, vector<KeyPoint> &out_keypoints,
                              OutputArray &descriptors)
{
   
    if (imageRaw.empty())
    {
        cerr << "image is empty! " << endl;
        return;
    }

    //为什么不直接在_image上操作?
    Mat image = imageRaw.getMat();
    //确保图片为单层灰度图像
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    //每一层分开保存
    vector<vector<KeyPoint>> allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);

    //对每一层的所有特征点计算描述子
    int nkeypoints = 0;

    cv::Mat descs;
    for(int level =0; level<_nlevels; level++)
    {
        nkeypoints += (int)allKeypoints[level].size();
    }
    if(nkeypoints==0)
    {
        descriptors.release();
    }
    else
    {
        descriptors.create(nkeypoints, 32, CV_8U);
        descs = descriptors.getMat();
    }

    out_keypoints.clear();
    out_keypoints.reserve(nkeypoints);

    int offset = 0;
    
    //为每层图像计算描述子
    for (int level = 0; level < _nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        Mat workingMat = _vImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        Mat desc = descs.rowRange(offset, offset + nkeypointsLevel);

        computeORBDesc(workingMat, keypoints, desc);

        offset += nkeypointsLevel;

        // 根据每层的尺度调整特征点的位置
        if (level != 0)
        {
            float scale = _vScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                 //keypoint乘以一个数就是调整x,y坐标了
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        out_keypoints.insert(out_keypoints.end(), keypoints.begin(), keypoints.end());
    }

}

void ORBextractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < _nlevels; ++level)
    {
        float scale = _vInvScaleFactor[level];
        Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        resize(image, _vImagePyramid[level], sz, 0, 0, INTER_LINEAR);
    }
}

//为每一层计算特征点并且进行均匀分布
void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>> &allKeypoints)
{

    allKeypoints.resize(_nlevels);
    //每个窗口应该是边长为30个像素的正方形
    const float W = 30;

    for (int level = 0; level < _nlevels; level++)
    {
        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = EDGE_THRESHOLD;
        const int maxBorderX = _vImagePyramid[level].cols - minBorderX;
        const int maxBorderY = _vImagePyramid[level].rows - minBorderY;

        vector<cv::KeyPoint> vToDistributeKeys;

        //有必要弄成10倍大小么? 每层的特征点数肯定不会超过nfeatures
        vToDistributeKeys.reserve(_nfeatures);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        //图片可以分成nCols*nRows个30*30的小格
        //但是因为取得是整形,小数点后被忽略了,所以其实图片可能去nCols.5*nRows.02个小格
        const int nCols = width / W;
        const int nRows = height / W;

        //重新计算每个小格的边长,并将结果向上取整,这样wCell*nCols会超出图片有效区域
        //需要在后续程序判断是否超出边界
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        //对每个窗口从上往下,从左往右的求特征点.
        //主要目的是通过调整FAST的阈值,来基本保证图片的每个区域里都有被检测到的特征点
        //而之后的OctTree是保证特征点之间的距离不要太近
        for (int i = 0; i < nRows; i++)
        {
            //确定小格的上下边界,以及判断是否超出有效区域
            //为什么ORB里面用的是float型?
            const int iniY = minBorderY + i * hCell;
            int maxY = iniY + hCell;
            if (iniY >= maxBorderY)
            {
                continue;
            }
            //由于wCell和hCell向上取整,所以要判断是否超出边界
            if (maxY > maxBorderY)
            {
                maxY = maxBorderY;
            }

            for (int j = 0; j < nCols; j++)
            {
                const int iniX = minBorderX + j * wCell;
                int maxX = iniX + wCell;
                if (iniX >= maxBorderX)
                {
                    continue;
                }
                if (maxX > maxBorderX)
                {
                    maxX = maxBorderX;
                }

                vector<cv::KeyPoint> vKeysCell;
                FAST(_vImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                     vKeysCell, _iniThFAST, true);

                if (vKeysCell.empty())
                {
                    FAST(_vImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell, _minThFAST, true);
                }

                if (!vKeysCell.empty())
                {
                    for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        //修正每个特征点在实际图片中的位置
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        //把当前所有检测到的FAST都放到该层待分布的关键点里
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }
        //开始对该层所有特征点进行平均分布
        vector<KeyPoint> &keypoints = allKeypoints[level];
        keypoints.reserve(_nfeatures);
        //对金字塔该层的特征点进行平均分布
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY, _nFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE * _vScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < _nlevels; ++level)
    {
        //直接把角度保存在了keypoint.angle里面了
        computeAngle(_vImagePyramid[level], allKeypoints[level]);
    }
  
}

vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                     const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    //图片的宽度除以图片的高度,这样创建一个初始的节点数量,
    //这样分出来的每个node接近正方形,一般图片都是宽度大于高度
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));
    //每个节点的宽度是多少
    const float hX = static_cast<float>(maxX - minX) / nIni;

    //用来保存所有节点的变量
    list<ExtractorNode> lNodes;

    //初始化节点
    vector<ExtractorNode *> vpIniNodes;
    vpIniNodes.resize(nIni);

    //对每一个初始节点设定边界
    //设定ExtractorNode的四个顶点,然后预留一部分内存
    for (int i = 0; i < nIni; i++)
    {
        ExtractorNode ni;
        ni._UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni._UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni._BL = cv::Point2i(ni._UL.x, maxY - minY);
        ni._BR = cv::Point2i(ni._UR.x, maxY - minY);
        ni._vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //把上一步提取到的该层图像的每一个特征点, 按照X轴坐标存放到对应的初始节点中.
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->_vKeys.push_back(kp);
    }

    //这步开始lNodes里面只是存放了初始节点
    list<ExtractorNode>::iterator lit = lNodes.begin();

    //看看是否所有的初始节点中都有特征点,把没有特征点的删除
    while (lit != lNodes.end())
    {
        if (lit->_vKeys.size() == 1)
        { //如果只有一个特征点,那么bNoMore就为真
            lit->_bNoMore = true;
            lit++;
        }
        else if (lit->_vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }
    //以上完成对初始节点的处理, 接下来开始分裂

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        //判断lNodes里面有多少是需要分裂的
        int nToExpand = 0;
        vSizeAndPointerToNode.clear();
        //开始遍历lNodes里面的所有的节点, 判断是否需要分裂
        //最初是几个初始节点
        while (lit != lNodes.end())
        {
            if (lit->_bNoMore)
            {
                lit++;
                continue;
            }
            else
            {
                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                //判断新分裂的四个节点是否包含特征点
                if (n1._vKeys.size() > 0)
                { //添加新节点到lNodes的最前面
                    //新分裂出来的节点放在list前面,而应该在这轮分裂的节点则放在list后面
                    lNodes.push_front(n1);
                    if (n1._vKeys.size() > 1)
                    { //如果超过一个特征点,则ToExpand增加一
                        nToExpand++;
                        //保存了特征点的数量和节点
                        vSizeAndPointerToNode.push_back(make_pair(n1._vKeys.size(), &lNodes.front()));
                        //就是n1的lit指向它自己
                        //这样n1到n4下来, lNodes最前面的节点的iterator永远指向自己
                        //没发现有什么用
                        lNodes.front()._lit = lNodes.begin();
                    }
                }
                if (n2._vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2._vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2._vKeys.size(), &lNodes.front()));
                        lNodes.front()._lit = lNodes.begin();
                    }
                }
                if (n3._vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3._vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3._vKeys.size(), &lNodes.front()));
                        lNodes.front()._lit = lNodes.begin();
                    }
                }
                if (n4._vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4._vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4._vKeys.size(), &lNodes.front()));
                        lNodes.front()._lit = lNodes.begin();
                    }
                }
                //分裂后将老的节点从list中删除, list只保存OctTree的叶子
                //erase之后lit指向下一个元素,如果没有下一个元素则指向end()
                lit = lNodes.erase(lit);
                //这个continue加不加没区别
                continue;
            }
        }
        //针对list上所有老的node分裂结束
        //老节点被从list上删除,现在list上存放的全部是新分裂出来的节点

        // 当全部节点的数量大于等于最大值
        //或者每个节点都只包含一个特征点时
        //分裂结束
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
        {
            bFinish = true;
        }
        //否则判断当list中的全部节点再分裂一次的话,节点总数会不会超过最大值
        //如果为yes,则按以下步骤完成后续分裂
        //否则返回上面一个while正常分裂.
        //下面这句话假设每个node都再分裂4个node出来. 这样可以保证特征点过密的区域不会分裂很多个node出来
        else if (((int)lNodes.size() + nToExpand * 3) > N)
        {

            while (!bFinish)
            {

                prevSize = lNodes.size();
                //所以从上一个while新分裂出来的包含多于一个特征点的节点和该节点保存的特征点数量
                //全部是节点的指针
                vector<pair<int, ExtractorNode *>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();
                //这是按照什么进行排序的? 可能是按照每个节点包含feature的数量
                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    ExtractorNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1._vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1._vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1._vKeys.size(), &lNodes.front()));
                            lNodes.front()._lit = lNodes.begin();
                        }
                    }
                    if (n2._vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2._vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2._vKeys.size(), &lNodes.front()));
                            lNodes.front()._lit = lNodes.begin();
                        }
                    }
                    if (n3._vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3._vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3._vKeys.size(), &lNodes.front()));
                            lNodes.front()._lit = lNodes.begin();
                        }
                    }
                    if (n4._vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4._vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4._vKeys.size(), &lNodes.front()));
                            lNodes.front()._lit = lNodes.begin();
                        }
                    }
                    //每个node保存一个指向自己的lit就是为了在这一步可以删除该node
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->_lit);
                    //和上一个while不同的是,这里每分裂一次就判断是否超过数量
                    if ((int)lNodes.size() >= N)
                        break;
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    //非极大值抑制
    //通过response来判断拿个FAST角点比较好
    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(_nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->_vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;

}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    //求出四个网格的中心点
    const int halfX = ceil(static_cast<float>(_UR.x - _UL.x) / 2);
    const int halfY = ceil(static_cast<float>(_BR.y - _UL.y) / 2);

    //Define boundaries of childs
    n1._UL = _UL;
    n1._UR = cv::Point2i(_UL.x + halfX, _UL.y);
    n1._BL = cv::Point2i(_UL.x, _UL.y + halfY);
    n1._BR = cv::Point2i(_UL.x + halfX, _UL.y + halfY);
    n1._vKeys.reserve(_vKeys.size());

    n2._UL = n1._UR;
    n2._BL = n1._BR;
    n2._BR = cv::Point2i(_UR.x, _UL.y + halfY);
    n2._vKeys.reserve(_vKeys.size());

    n3._UL = n1._BL;
    n3._UR = n1._BR;
    n3._BL = _BL;
    n3._BR = cv::Point2i(n1._BR.x, _BL.y);
    n3._vKeys.reserve(_vKeys.size());

    n4._UL = n3._UR;
    n4._UR = n2._BR;
    n4._BL = n3._BR;
    n4._BR = _BR;
    n4._vKeys.reserve(_vKeys.size());

    //往网格里面填特征点
    //Associate points to childs
    for (size_t i = 0; i < _vKeys.size(); i++)
    {
        const cv::KeyPoint &kp = _vKeys[i];
        if (kp.pt.x < n1._UR.x)
        {
            if (kp.pt.y < n1._BR.y)
                n1._vKeys.push_back(kp);
            else
                n3._vKeys.push_back(kp);
        }
        else if (kp.pt.y < n1._BR.y)
            n2._vKeys.push_back(kp);
        else
            n4._vKeys.push_back(kp);
    }
    //判断是否只剩下一个特征点了
    if (n1._vKeys.size() == 1)
        n1._bNoMore = true;
    if (n2._vKeys.size() == 1)
        n2._bNoMore = true;
    if (n3._vKeys.size() == 1)
        n3._bNoMore = true;
    if (n4._vKeys.size() == 1)
        n4._bNoMore = true;
}