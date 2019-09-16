#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include <memory>
using namespace cv;
using namespace std;


class ExtractorNode
{
public:
    ExtractorNode() : _bNoMore(false) {}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> _vKeys;
    cv::Point2i _UL, _UR, _BL, _BR;
    std::list<ExtractorNode>::iterator _lit;
    bool _bNoMore;
};

class ORBextractor
{   
    public:
    using Ptr = std::shared_ptr<ORBextractor>;
    using ConstPtr = std::shared_ptr<const ORBextractor>;

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,int iniThFAST, int minThFAST);

    void ComputePyramid(cv::Mat image);

    void operator()( InputArray image, vector<KeyPoint> &keypoints,
                      OutputArray &descriptors);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints); 

    vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level);

    std::vector<cv::Mat> _vImagePyramid;

    int _nfeatures;
    double _scaleFactor;
    int _nlevels;
    int _iniThFAST;
    int _minThFAST;

    std::vector<int> _nFeaturesPerLevel;

    std::vector<int> _umax;

    std::vector<float> _vScaleFactor;
    std::vector<float> _vInvScaleFactor;
    std::vector<float> _vLevelSigma2;
    std::vector<float> _vInvLevelSigma2;
};

#endif