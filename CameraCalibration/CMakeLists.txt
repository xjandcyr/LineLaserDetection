
cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME camera_calibration)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_COMPILER "D:/openCV/mingw64-8.1.0/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "D:/openCV/mingw64-8.1.0/bin/g++.exe")
set(OpenCV_DIR "D:/openCV/OpenCV-MinGW-Build-4.5.2/x64/mingw/lib")

find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# 主程序
add_executable(${PROJECT_NAME}
       src/main.cpp
       src/camera_functions.cpp
       src/roi_selector.cpp
   )

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# 四边形ROI测试程序
add_executable(test_quad_roi
       src/test_quad_roi.cpp
       src/roi_selector.cpp
   )

target_link_libraries(test_quad_roi PRIVATE ${OpenCV_LIBS})
