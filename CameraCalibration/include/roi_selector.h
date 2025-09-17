#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 四边形ROI结构体
struct QuadROI {
    std::vector<cv::Point2f> points;  // 四个顶点
    cv::Rect boundingBox;             // 外接矩形
    
    QuadROI() : points(4), boundingBox(0, 0, 0, 0) {}
    QuadROI(const std::vector<cv::Point2f>& pts) : points(pts) {
        updateBoundingBox();
    }
    
    void updateBoundingBox() {
        if (points.size() != 4) return;
        
        float minX = points[0].x, maxX = points[0].x;
        float minY = points[0].y, maxY = points[0].y;
        
        for (const auto& pt : points) {
            minX = std::min(minX, pt.x);
            maxX = std::max(maxX, pt.x);
            minY = std::min(minY, pt.y);
            maxY = std::max(maxY, pt.y);
        }
        
        boundingBox = cv::Rect(minX, minY, maxX - minX, maxY - minY);
    }
    
    bool isValid() const {
        return points.size() == 4 && boundingBox.width > 0 && boundingBox.height > 0;
    }
};

// 矩形ROI选择（原有功能）
cv::Rect selectROIFromImage(const cv::Mat& image, const std::string& windowName = "选择ROI");

// 四边形ROI选择（新功能）
QuadROI selectQuadROIFromImage(const cv::Mat& image, const std::string& windowName = "选择四边形ROI");

// 保存ROI区域到文件
bool saveROIToFile(const cv::Rect& roi, const std::string& filePath);
bool saveQuadROIToFile(const QuadROI& quadRoi, const std::string& filePath);

// 从文件加载ROI区域
bool loadROIFromFile(cv::Rect& roi, const std::string& filePath);
bool loadQuadROIFromFile(QuadROI& quadRoi, const std::string& filePath);

// 绘制四边形ROI
void drawQuadROI(cv::Mat& image, const QuadROI& quadRoi, const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness = 2);

// 检查点是否在四边形内
bool isPointInQuadROI(const cv::Point2f& point, const QuadROI& quadRoi); 