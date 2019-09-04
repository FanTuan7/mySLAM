#include "dataset.h"

//设定文件路径,读取相机参数
Dataset::Dataset(const std::string &dataset_path)
{
    _left_images_path_start = dataset_path + "image_0/";
    _right_images_path_start = dataset_path + "image_1/";
    _image_path_end = ".png";
    _times_file_path = dataset_path + "times.txt";
    _camera_file_path = dataset_path + "calib.txt";

    //读取相机数据
    std::ifstream fCalib;
    fCalib.open(_camera_file_path.c_str());
    if (!fCalib.is_open())
    {
        std::cerr << "can't open calib.txt " << std::endl;
        return;
    }

    //四个相机
    for (int i = 0; i < 4; i++)
    {
        char camera_name[3];
        for (int j = 0; j < 3; j++)
        {
            fCalib >> camera_name[j];
        }

        double projection_matrix[12];
        for (int j = 0; j < 12; j++)
        {
            fCalib >> projection_matrix[j];
        }

        int id = (int)camera_name[1] - '0';
        double fx = projection_matrix[0];
        double fy = projection_matrix[5];
        double cx = projection_matrix[2];
        double cy = projection_matrix[6];
        double fb = abs(projection_matrix[3]);

        Camera::Ptr new_camera = Camera::Ptr(new Camera{id,fx,fy,cx,cy,fb});

        _cameras.push_back(new_camera);
    }
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
        std::cerr << "load image " <<  _current_image_index << " failed" << std::endl;
        return nullptr;
    }
    //待解决问题: 当图片读取完时干什么?
    Frame::Ptr newFrame(new Frame(_current_image_index,img_left, img_right));
    //左右相机的数据都一样
    newFrame->_camera = _cameras[1];
    _current_image_index++;

    return newFrame;
}

Camera::Ptr Dataset::getCamera(int camera_id)
{
    return _cameras[camera_id];
}

void Dataset::loadDBoW3(std::string path)
{
    _vocab = DBoW3::Vocabulary(path);

    if (_vocab.empty()) 
    {
        cerr << "set vocabulary falied." << endl;
    }
}