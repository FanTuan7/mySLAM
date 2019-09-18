#ifndef HELPER_H
#define HELPER_H
//#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <iostream>
#include <vector>
#include <chrono>

#include <sophus/se3.hpp>

using namespace std;
//using namespace Eigen;

/*inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]);
}
*/
//从2到1的变换
void pose_estimation_3d3d (
    const vector<Eigen::Vector3d>& pts1,
    const vector<Eigen::Vector3d>& pts2,
    Eigen::Matrix3d &R, Eigen::Vector3d &t
);
/*
void pose_estimation_BA (
    vector<Eigen::Vector3d>& pts1,
    vector<Eigen::Vector3d>& pts2,
    Eigen::Matrix3d &R, Eigen::Vector3d &t
);
*/
#endif