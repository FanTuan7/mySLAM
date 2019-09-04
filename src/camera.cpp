#include "camera.h"


Camera::Camera(int id, double fx, double fy, double cx, double cy, double fb)
:_id(id),_fx(fx),_fy(fy),_cx(cx),_cy(cy),_fb(fb)
{

}

Eigen::Matrix3f Camera::K()
{
    Eigen::Matrix3f k;
    k << _fx,  0,_cx,
           0,_fy,_cy,
           0,  0,  1;
    return k;
}