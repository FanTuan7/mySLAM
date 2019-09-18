cmake_minimum_required(VERSION 2.8)
Project(mySLAM)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Debug") 
#set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

add_executable(main example/main.cpp)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


find_package(Boost  REQUIRED )
Find_Package(Cholmod REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(Eigen3 REQUIRED)
Find_Package(g2o REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
#find_package(yaml-cpp REQUIRED )


include_directories(${PROJECT_SOURCE_DIR}/include 
                    ${Boost_INCLUDE_DIR} 
                    ${CHOLMOD_INCLUDE_DIR}
                    ${DBoW3_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR} 
                    ${G2O_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS} 
                    ${Pangolin_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS} 
                   # ${YAML_CPP_INCLUDE_DIR} 
)

add_library(${PROJECT_NAME} SHARED
src/camera.cpp
src/dataset.cpp
src/frame.cpp
src/helper.cpp
#src/local_mapping_custom_type.cpp
src/local_mapping_g2o_type.cpp
src/map_point.cpp
src/map.cpp
src/ORBextractor.cpp
src/ORBmatcher.cpp
src/tracking.cpp
src/viewer.cpp
)


SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
    g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

target_link_libraries(main  ${PROJECT_NAME}
                            ${Boost_LIBS} 
                            ${CHOLMOD_LIBRARIES}
                            ${DBoW3_LIBS}
                            ${EIGEN3_LIBS}
                            ${G2O_LIBS} 
                            ${OpenCV_LIBS} 
                            ${Pangolin_LIBRARIES} #换成LIBS就会出错
                            ${Sophus_LIBS}
                            #直接添加yaml-cpp的静态库 需要根据你的环境修改
                            /home/fantuan/tools/yaml-cpp/build/libyaml-cpp.a
                            #${YAML_CPP_LIBS}
)

