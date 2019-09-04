
#ifndef DATASET_H
#define DATASET_H

#include "camera.h"
#include "frame.h"
#include <memory>
#include <DBoW3/DBoW3.h>

class Dataset
{
public:
    //测试下不加这句话能会发生什么
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Dataset>;
    using ConstPtr = std::shared_ptr<const Dataset>;

    Dataset(const std::string &dataset_path);

    bool Init();

    Frame::Ptr NextFrame();
    // 0 1 2 3 灰左 灰右 彩左 彩右
    Camera::Ptr getCamera(int camera_id);



    std::string _left_images_path_start ;
    std::string _right_images_path_start;
    std::string _image_path_end;
    std::string _times_file_path;
    std::string _camera_file_path;

    long _current_image_index = 0;

    std::vector<Camera::Ptr> _cameras;


    DBoW3::Vocabulary _vocab;
    void loadDBoW3(std::string path);

};


#endif