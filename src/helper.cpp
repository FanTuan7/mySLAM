#include "helper.h"


using namespace std;
//using namespace Eigen;
//从2到1的变换
void pose_estimation_3d3d (
    const vector<Eigen::Vector3d>& pts1,
    const vector<Eigen::Vector3d>& pts2,
    Eigen::Matrix3d &R, Eigen::Vector3d &t
)
{
    Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d p2 = Eigen::Vector3d::Zero();
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1/=N ;
    p2/=N;
    vector<Eigen::Vector3d>  q1(N), q2(N); 
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += q2[i] * q1[i].transpose();
    }

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    if (U.determinant() * V.determinant() < 0)
	{
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
	}

    R = U * (V.transpose());
    t = p2 - R * p1;

    /*if (U.determinant() * V.determinant() < 0)
	{
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
	}

    Eigen::Matrix3d R_ = U* ( V.transpose() );
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
        );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );*/
}

