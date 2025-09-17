
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(LidarLineDetection)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_COMPILER "D:/openCV/mingw64-8.1.0/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "D:/openCV/mingw64-8.1.0/bin/g++.exe")
set(OpenCV_DIR "D:/openCV/OpenCV-MinGW-Build-4.5.2/x64/mingw/lib")

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)


# 添加头文件搜索路径
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
)

# 添加可执行文件（确保实现文件也加入）
add_executable(TestLidarLineDetection src/lidar_test_main.cpp src/lidar_line_detection.cpp src/camera_stability_detection.cpp)

# 添加四边形ROI测试程序
add_executable(QuadROITest src/quad_roi_test.cpp src/lidar_line_detection.cpp src/camera_stability_detection.cpp)

# 添加共享库
add_library(LidarLineDetection SHARED src/lidar_line_detection.cpp src/camera_stability_detection.cpp)

# 链接库
# TestLidarLineDetection 只需链接 OpenCV
# QuadROITest 需要链接 OpenCV
# LidarLineDetection 需要链接 OpenCV
# 如果你希望TestLidarLineDetection只测试主程序，也可以不链接库
# 但如果要测试动态库接口，则保留下面的链接

target_link_libraries(TestLidarLineDetection ${OpenCV_LIBS})
target_link_libraries(QuadROITest ${OpenCV_LIBS})
target_link_libraries(LidarLineDetection ${OpenCV_LIBS})



# cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=D:/openCV/mingw64-8.1.0/bin/gcc.exe -DCMAKE_CXX_COMPILER=D:/openCV/mingw64-8.1.0/bin/g++.exe -DOpenCV_DIR=D:/openCV/OpenCV-MinGW-Build-4.5.2/x64/mingw/lib ..
# mingw32-make