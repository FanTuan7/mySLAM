#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>
#include <stdlib.h>

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"

#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class VertexCamera : public g2o::BaseVertex<6, Eigen::Matrix<double, 6, 1>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexCamera() {}

    virtual bool read(std::istream &) { return false; }
    virtual bool write(std::ostream &) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update)
    {
        //本程序默认外参顺序符合Sophus库，先平移，再旋转。
        Sophus::Vector6d se3_estimate;
        se3_estimate << _estimate[0], _estimate[1], _estimate[2], _estimate[3], _estimate[4], _estimate[5];

        Sophus::Vector6d se3_update;
        se3_update << update[0], update[1], update[2], update[3], update[4], update[5];

        //update se3
        se3_estimate = (Sophus::SE3d::exp(se3_update) * Sophus::SE3d::exp(se3_estimate)).log();
        Eigen::Matrix<double, 6, 1> x;

        x << se3_estimate[0], se3_estimate[1], se3_estimate[2],
             se3_estimate[3], se3_estimate[4], se3_estimate[5];

        _estimate = x;
    }
};

class Vertex3DPoint : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Vertex3DPoint() {}

    virtual bool read(std::istream &) { return false; }
    virtual bool write(std::ostream &) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update)
    {
        Eigen::Vector3d::ConstMapType x(update);
        _estimate += x;
    }
};

class EdgeCamera_3Dpoint : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCamera, Vertex3DPoint>
{
public:
    float fx = 718.856;
    float fy = 718.856;
    float cx = 607.1928;
    float cy = 185.2157;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeCamera_3Dpoint(){};

    virtual bool read(std::istream &) { return false; }

    virtual bool write(std::ostream &) const { return false; }

    virtual void computeError() override
    {
        const VertexCamera *vertex_c_ptr = static_cast<const VertexCamera *>(vertex(0));
        const Vertex3DPoint *vertex_p_ptr = static_cast<const Vertex3DPoint *>(vertex(1));

        Eigen::Matrix<double, 6, 1> c_param = vertex_c_ptr->estimate();
        Sophus::Vector6d se3;
        se3 << c_param(0), c_param(1), c_param(2), c_param(3), c_param(4), c_param(5);
        Sophus::SE3d SE3 = Sophus::SE3d::exp(se3);

        Eigen::Vector3d P_w, P_c, p;

        P_w = vertex_p_ptr->estimate();

        P_c = SE3.rotationMatrix() * P_w + SE3.translation();

        p(0) = P_c(0) / P_c(2);
        p(1) = P_c(1) / P_c(2);

        double u = fx * p(0) + cx;
        double v = fx * p(1) + cx;

        _error[0] = _measurement[0] - u;
        _error[1] = _measurement[1] - v;
    }

    /*virtual void linearizeOplus() override
    {

        const VertexCamera *vertex_c_ptr = static_cast<const VertexCamera *>(vertex(0));
        const Vertex3DPoint *vertex_p_ptr = static_cast<const Vertex3DPoint *>(vertex(1));

        Eigen::Matrix<double, 6, 1> c_param = vertex_c_ptr->estimate();
        Sophus::Vector6d se3;
        se3 << c_param(0), c_param(1), c_param(2), c_param(3), c_param(4), c_param(5);
        Sophus::SE3d SE3 = Sophus::SE3d::exp(se3);

        Eigen::Vector3d P_w = vertex_p_ptr->estimate();
        Eigen::Vector3d P_c = SE3.rotationMatrix() * P_w + SE3.translation();

        double x = P_c[0];
        double y = P_c[1];
        double z = P_c[2];

        Eigen::Matrix<double, 2, 3> de_dP_c;
        de_dP_c(0, 0) = -fx / z;
        de_dP_c(0, 1) = 0;
        de_dP_c(0, 2) = fx * x / z / z;

        de_dP_c(1, 0) = 0;
        de_dP_c(1, 1) = -fy / z;
        de_dP_c(1, 2) = fy * y / z / z;

        Eigen::Matrix<double, 3, 6> dP_c_dSE3delta;
        dP_c_dSE3delta << 1, 0, 0, 0, z, -y,
            0, 1, 0, -z, 0, x,
            0, 0, 1, y, -x, 0;

        Eigen::Matrix<double, 2, 6> J_extrinsic;
        J_extrinsic = de_dP_c * dP_c_dSE3delta;

        Eigen::Matrix<double, 2, 3> J_P_w;
        J_P_w = de_dP_c * SE3.rotationMatrix();

        _jacobianOplusXi(0, 0) = J_extrinsic(0, 0);
        _jacobianOplusXi(0, 1) = J_extrinsic(0, 1);
        _jacobianOplusXi(0, 2) = J_extrinsic(0, 2);
        _jacobianOplusXi(0, 3) = J_extrinsic(0, 3);
        _jacobianOplusXi(0, 4) = J_extrinsic(0, 4);
        _jacobianOplusXi(0, 5) = J_extrinsic(0, 5);
        _jacobianOplusXi(1, 0) = J_extrinsic(1, 0);
        _jacobianOplusXi(1, 1) = J_extrinsic(1, 1);
        _jacobianOplusXi(1, 2) = J_extrinsic(1, 2);
        _jacobianOplusXi(1, 3) = J_extrinsic(1, 3);
        _jacobianOplusXi(1, 4) = J_extrinsic(1, 4);
        _jacobianOplusXi(1, 5) = J_extrinsic(1, 5);

        _jacobianOplusXj(0, 0) = J_P_w(0, 0);
        _jacobianOplusXj(0, 1) = J_P_w(0, 1);
        _jacobianOplusXj(0, 2) = J_P_w(0, 2);
        _jacobianOplusXj(1, 0) = J_P_w(1, 0);
        _jacobianOplusXj(1, 1) = J_P_w(1, 1);
        _jacobianOplusXj(1, 2) = J_P_w(1, 2);
    }*/
};
