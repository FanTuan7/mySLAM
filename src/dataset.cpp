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
    double fx = _config["fx"].as<double>();
    double fy = _config["fy"].as<double>();
    double cx = _config["cx"].as<double>();
    double cy = _config["cy"].as<double>();
    double fb = _config["fb"].as<double>();
    _camera = Camera::ConstPtr(new Camera{id, fx, fy, cx, cy, fb});

    _features = _config["features"].as<int>();
    _scaleFactor = _config["scaleFactor"].as<float>();
    _levels = _config["levels"].as<int>();
    _iniThFAST = _config["iniThFAST"].as<int>();
    _minThFAST = _config["minThFAST"].as<int>();

    _KF_DoWrate_Low = _config["KF_DoWrate_Low"].as<float>();
    _KF_DoWrate_High  = _config["KF_DoWrate_High"].as<float>();
    _KF_mindistance = _config["KF_mindistance"].as<int>();
    _KF_maxdistance = _config["KF_maxdistance"].as<int>();


    vocab_path = _config["Vocabulary"].as<std::string>();

    _groundtruth_path  = _config["Groundtruth"].as<std::string>();
    getGroundtruth();

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
}

//viewr里面画点是用的Eigen:Vector3d 数据. 所以这里只需要存储相机的translation就行
void Dataset::getGroundtruth()
{
    ifstream fin(_groundtruth_path);

    std::vector<double> numbers;

    if(! fin) 
    {
        std::cerr << "不能读取 ground truth。 文件打不开\n"; 
    }

    std::string line;
    while(getline(fin, line))
    {
        std::istringstream is(line);
        numbers = std::vector<double>( std::istream_iterator<double>(is),
                              std::istream_iterator<double>() ) ;

        _groundtruth.push_back(Eigen::Vector3d(numbers[3],numbers[7],numbers[11]));
    }
      
}