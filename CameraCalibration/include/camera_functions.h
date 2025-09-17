#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct CameraCalibration {
    cv::Point2d target_center;
    double max_deviation = 5.0;
    bool saveToFile(const std::string& filePath) const;
    bool loadFromFile(const std::string& filePath);
};

bool detectTarget(const cv::Mat& image, std::vector<cv::Point2f>& corners);
cv::Point2d calculateTargetCenter(const std::vector<cv::Point2f>& corners);
bool detectLidarLine(const cv::Mat& inputImage, const cv::Rect& roiRect, cv::Vec4f& line);
bool cameraSelfCheck(const cv::Mat& image, const CameraCalibration& calibration, double& deviation); 