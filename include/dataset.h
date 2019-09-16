#ifndef DATASET_H
#define DATASET_H

#include "camera.h"
#include "frame.h"
#include <memory>
#include <yaml-cpp/yaml.h>
#include <DBoW3/DBoW3.h>

class Dataset
{
public:
    //测试下不加这句话能会发生什么
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Dataset>;
    using ConstPtr = std::shared_ptr<const Dataset>;

    Dataset(const std::string &yaml_file_path);

    bool Init();

    Frame::Ptr NextFrame();
    Camera::ConstPtr getCamera(); //修改,只返回一个相机

    std::string _left_images_path_start ;
    std::string _right_images_path_start;
    std::string _image_path_end;
    long _current_image_index = 0;

    
    YAML::Node _config;
    //以下参数都是要从yaml文件中读取的
    Camera::ConstPtr _camera;
    int _features;
    float _scaleFactor;
    int _levels;
    int _iniThFAST;
    int _minThFAST;

    float _KF_DoWrate;
    float _KF_mindistance;

    std::string vocab_path;
    DBoW3::Vocabulary _vocab;
    DBoW3::Database _db;
    void loadDBoW3();
};


#endif