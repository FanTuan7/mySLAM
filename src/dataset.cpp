#include "dataset.h"

//设定文件路径,读取相机参数
Dataset::Dataset(const std::string &yaml_file_path)
{
    _config = YAML::LoadFile(yaml_file_path);

    if (!_config["datafile"])
    {
        std::cerr << "can't load yaml file, please check the path" << endl;
        return;
    }

    std::string dataset_path = _config["datafile"].as<std::string>();
    _left_images_path_start = dataset_path + "image_0/";
    _right_images_path_start = dataset_path + "image_1/";
    _image_path_end = ".png";

    //读取相机参数
    int id = 0;
    float fx = _config["fx"].as<float>();
    float fy = _config["fy"].as<float>();
    float cx = _config["cx"].as<float>();
    float cy = _config["cy"].as<float>();
    float fb = _config["fb"].as<float>();
    _camera = Camera::ConstPtr(new Camera{id, fx, fy, cx, cy, fb});

    _features = _config["features"].as<int>();
    _scaleFactor = _config["scaleFactor"].as<float>();
    _levels = _config["levels"].as<int>();
    _iniThFAST = _config["iniThFAST"].as<int>();
    _minThFAST = _config["minThFAST"].as<int>();

    _KF_DoWrate = _config["KF_DoWrate"].as<float>();
    _KF_mindistance = _config["KF_mindistance"].as<float>();


    vocab_path = _config["Vocabulary"].as<std::string>();
}

Frame::Ptr Dataset::NextFrame()
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << _current_image_index;
    std::string left_image_path = _left_images_path_start + ss.str() + _image_path_end;
    std::string right_image_path = _right_images_path_start + ss.str() + _image_path_end;

    cv::Mat img_left, img_right;
    img_left = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
    img_right = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

    if (img_left.empty() || img_right.empty())
    {
        std::cerr << "load image " << _current_image_index << " failed" << std::endl;
        return nullptr;
    }
    //待解决问题: 当图片读取完时干什么?
    Frame::Ptr newFrame(new Frame(_current_image_index, img_left, img_right));
    //左右相机的数据都一样
    newFrame->_camera = _camera;
    _current_image_index++;

    return newFrame;
}

Camera::ConstPtr Dataset::getCamera()
{
    return _camera;
}

void Dataset::loadDBoW3()
{
    _vocab = DBoW3::Vocabulary(vocab_path);

    if (_vocab.empty())
    {
        cerr << "set vocabulary falied." << endl;
    }
    //_db(_vocab, false, 0);
}