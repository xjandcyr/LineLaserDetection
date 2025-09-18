#include <iostream>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include "lidar_line_detection.h"

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "四边形ROI激光线检测测试程序" << std::endl;

    // 读取测试图像
    cv::Mat testImage = cv::imread("D:\\OpenCV\\Code\\PV31\\image\\111.jpg");
    if (testImage.empty()) {
        std::cout << "[错误] 无法加载测试图像" << std::endl;
        return 1;
    }

    // 创建四边形ROI可视化图像
    cv::Mat visualizationImage = testImage.clone();
    
    // 定义四边形ROI的四个点
    std::vector<cv::Point> quadPoints = {
        cv::Point(322, 548),
        cv::Point(1601, 587),
        cv::Point(1604, 632),
        cv::Point(322, 619)
    };

    // 绘制四边形ROI
    for (int i = 0; i < 4; i++) {
        cv::line(visualizationImage, quadPoints[i], quadPoints[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
        cv::circle(visualizationImage, quadPoints[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(visualizationImage, "X" + std::to_string(i + 1), 
                   quadPoints[i] + cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
    }

    // 保存四边形ROI可视化图像
    cv::imwrite("D:\\OpenCV\\Code\\PV31\\output\\quad_roi_visualization.jpg", visualizationImage);
    std::cout << "四边形ROI可视化图像已保存: D:\\OpenCV\\Code\\PV31\\output\\quad_roi_visualization.jpg" << std::endl;

    // 测试四边形ROI激光线检测
    std::string roiConfigPath = "D:\\OpenCV\\Code\\PV31\\config\\ROI_CONFIG.ini";
    LidarLineDetector::LidarLineResult result = LidarLineDetector::detect(testImage, roiConfigPath, "QUAD_TEST", "D:/OpenCV/Code/PV31/output");
    
    if (result.error_code == DetectionResultCode::SUCCESS) {
        std::cout << "[四边形ROI激光线检测] 成功" << std::endl;
        std::cout << "  角度(弧度): " << result.line_angle << std::endl;
        std::cout << "  角度(度): " << result.line_angle * 180.0 / CV_PI << std::endl;
        std::cout << "  结果图像: " << result.image_path << std::endl;
    } else {
        std::cout << "[四边形ROI激光线检测] 失败，错误码: " << static_cast<int>(result.error_code) << std::endl;
        if (!result.image_path.empty()) {
            std::cout << "  失败图像: " << result.image_path << std::endl;
        }
    }

    return 0;
}