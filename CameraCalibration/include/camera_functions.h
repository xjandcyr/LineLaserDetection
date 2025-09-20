#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 我的窗口大小是1980 * 1080，同比例缩小
#define WINDOW_WIDTH 1980
#define WINDOW_HEIGHT 1080

struct CameraCalibration {
    cv::Point2d target_center;
    double max_deviation = 5.0;
    bool savePointToFile(const std::string& filePath);
    bool loadFromFile(const std::string& filePath);
};

bool detectTarget(const cv::Mat& image, std::vector<cv::Point2f>& corners);
cv::Point2d calculateTargetCenter(const std::vector<cv::Point2f>& corners);
bool detectLidarLine(const cv::Mat& inputImage, const cv::Rect& roiRect, cv::Vec4f& line);
bool cameraSelfCheck(const cv::Mat& image, const CameraCalibration& calibration, double& deviation); 