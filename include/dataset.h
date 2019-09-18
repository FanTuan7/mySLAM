#ifndef DATASET_H
#define DATASET_H

#include "camera.h"
#include "frame.h"
#include <memory>
#include <yaml-cpp/yaml.h>
#include <DBoW3/DBoW3.h>
#include <Eigen/Core>

#include <fstream>// std::ifstream
#include <iostream>// std::wcout
#include <vector>
#include <string>

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
    Camera::ConstPtr getCamera(); 

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

    float _KF_DoWrate_Low;
    float _KF_DoWrate_High;
    int _KF_mindistance;
    int _KF_maxdistance;


    std::string vocab_path;
    DBoW3::Vocabulary _vocab;
    DBoW3::Database _db;
    void loadDBoW3();

    void getGroundtruth();
    std::string _groundtruth_path;
    std::vector<Eigen::Vector3d> _groundtruth;
};


#endif