#include "config/config.h"

void Config::setParameterFile(const std::string& filename)
{
    if(_config == nullptr)
    {
        _config = std::shared_ptr<Config>(new Config);
    }

    _config->_file = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);

    if(_config->_file.isOpened() ==false)
    {
        std::cerr << "parameter file " << filename << " dose not exist. " << std::endl;
        _config->_file.release();
        return;
    }
}

Config::~Config()
{
    if(_file.isOpened())
    {
        _file.release();
    }
}


std::shared_ptr<Config> Config::_config = nullptr;

