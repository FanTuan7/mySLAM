#include <memory>
#include <opencv2/opencv.hpp>

class Config
{
    private:
    static std::shared_ptr<Config> _config;
    cv::FileStorage _file;

    Config(){} //singleton

    public:
    ~Config();

    static void setParameterFile(const std::string& filename);

    template <typename T>
    static T get(const std::string& key)
    {
        return T(Config::_config->_file[key]);
    }
};

//使用示范
/*
Config::setParameterFile(".yaml");
double fx = Config::get<double> ("Camera.fx");
*/
