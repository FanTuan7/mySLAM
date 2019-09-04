cmake_minimum_required(VERSION 2.8)
Project(mySLAM)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Debug") 
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")


add_library(${PROJECT_NAME} SHARED
src/camera.cpp
src/frame.cpp
src/ORBextractor.cpp
src/ORBmatcher.cpp
src/dataset.cpp
src/tracking.cpp
src/helper.cpp
src/map_point.cpp
src/map.cpp
src/viewer.cpp
)


set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost  REQUIRED )
include_directories( ${PROJECT_SOURCE_DIR}/include ${DBoW3_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/config ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS})



add_executable(main src/main.cpp)

target_link_libraries(main 
${PROJECT_NAME}
${Pangolin_LIBRARIES} 
${OpenCV_LIBS} 
${EIGEN3_LIBS} 
${Sophus_LIBRARIES}
${DBoW3_LIBS}
)

