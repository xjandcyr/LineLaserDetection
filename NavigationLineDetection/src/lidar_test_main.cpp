#include <iostream>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include "lidar_line_detection.h"
#include <iomanip> // Required for std::fixed and std::setprecision
#include <ctime> // Required for time()

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "激光线检测库 v" << LidarLineDetector::getVersionMajor() << "."
              << LidarLineDetector::getVersionMinor() << "."
              << LidarLineDetector::getVersionPatch() << std::endl;

    // 激光线检测
    cv::Mat testImage = cv::imread("../image/111.jpg");
    if (testImage.empty()) {
        std::cout << "[错误] 无法加载激光线测试图像" << std::endl;
        return 1;
    }
    
    // 直接调用新版接口，自动处理四边形ROI配置
    std::string roiConfigPath = "../config/NavLineCheckRoi.yml";
    LidarLineDetector::LidarLineResult result = LidarLineDetector::lineDetect(testImage, roiConfigPath, "123456", "../output", 5);

    if (result.error_code == DetectionResultCode::SUCCESS) {
        std::cout << "[激光线检测] sucess: " << static_cast<int>(result.error_code)<< std::endl;
    } else {
        std::cout << "[激光线检测] 失败，错误码: " << static_cast<int>(result.error_code);
        if (!result.image_path.empty()) {
            std::cout << "\n  失败图像: " << result.image_path;
        }
        std::cout << std::endl;
    }

    // 相机移动检测
    cv::Mat cameraImage = cv::imread("../image/111.jpg");
    if (cameraImage.empty()) {
        std::cout << "[错误] 无法加载相机测试图像" << std::endl;
        return 1;
    }
    // 新接口：直接传入配置文件路径和输出目录，图片保存直接在函数内部进行
    std::string centerPointConfigPath = "../config/SelfCheckCenterPoint.yml";
    std::string cameraSelfCheckRoiConifgPath = "../config/CameraSelfCheckRoi.yml";
    TargetMovementResult_C moveResult = CameraStabilityDetection::checkCameraMovement(cameraImage, centerPointConfigPath, cameraSelfCheckRoiConifgPath, "../output");
    if (moveResult.error_code == static_cast<int>(DetectionResultCode::SUCCESS)) {
        std::cout << "[相机移动检测] " << (moveResult.is_stable ? "未移动" : "已移动")
                  << "，移动距离: " << std::fixed << std::setprecision(2) << moveResult.distance << " 像素" << std::endl;
        std::cout << "  检测结果: " << moveResult.message << std::endl;
    } else {
        std::cout << "[相机移动检测] 检测失败，错误码: " << moveResult.error_code << std::endl;
        std::cout << "  错误信息: " << moveResult.message << std::endl;
    }
    return 0;
}