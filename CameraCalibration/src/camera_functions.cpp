#include "camera_functions.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
using namespace cv;
using namespace std;

// CameraCalibration实现
bool CameraCalibration::savePointToFile(const string& filePath)
{
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "SelfCheckCenterPoint" << "{";
    fs << "center_x" << cv::format("%.2f", target_center.x);
    fs << "center_y" << cv::format("%.2f", target_center.y);
    fs << "}";
    fs.release();
    return true;
}

bool CameraCalibration::loadFromFile(const string& filePath) {
    try {
        ifstream file(filePath);
        if (!file.is_open()) return false;
        string line;
        bool xRead = false, yRead = false, devRead = false;
        while (getline(file, line)) {
            if (line.find("center_x:") == 0) {
                sscanf(line.c_str(), "center_x: %lf", &target_center.x);
                xRead = true;
            } else if (line.find("center_y:") == 0) {
                sscanf(line.c_str(), "center_y: %lf", &target_center.y);
                yRead = true;
            } else if (line.find("tolerance:") == 0) {
                sscanf(line.c_str(), "tolerance: %lf", &max_deviation);
                devRead = true;
            }
        }
        file.close();
        return xRead && yRead && devRead;
    } catch (const exception& e) {
        cerr << "加载标定文件失败: " << e.what() << endl;
        return false;
    }
}



// 检测标靶四个角落的黑色方块
bool detectTarget(const Mat& image, vector<Point2f>& corners) {
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 80, 255, THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    vector<Point2f> centers;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        
        // 4个黑色方块的面积都为9000左右
        if (area < 5000 || area > 13000)
        {
            continue;
        }

        Rect rect = boundingRect(contour);
        double aspect = (double)rect.width / rect.height;

        if (aspect > 0.8 && aspect < 1.2) {
            Moments m = moments(contour);
            if (m.m00 != 0) {
                centers.push_back(Point2f(m.m10/m.m00, m.m01/m.m00));
            }
        }
    }
    
    if (centers.size() != 4) {
        cerr << "未能检测到4个标靶方块，找到: " << centers.size() << endl;
        return false;
    }
    sort(centers.begin(), centers.end(), [](const Point2f& a, const Point2f& b) {
        return a.y < b.y || (a.y == b.y && a.x < b.x);
    });
    if (centers[0].x > centers[1].x) swap(centers[0], centers[1]);
    if (centers[2].x > centers[3].x) swap(centers[2], centers[3]);
    corners = centers;
    return true;
}

// 计算标靶中心点
Point2d calculateTargetCenter(const vector<Point2f>& corners) {
    if (corners.size() != 4) return Point2d(-1, -1);
    double centerX = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4.0;
    double centerY = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4.0;
    return Point2d(centerX, centerY);
}

// 提取ROI区域并检测激光线
bool detectLidarLine(const Mat& inputImage, const Rect& roiRect, Vec4f& line) {
    Mat roi = inputImage(roiRect).clone();
    Mat gray, blurred, edges;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    Canny(blurred, edges, 50, 150);
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
    if (lines.empty()) {
        cout << "未检测到直线" << endl;
        return false;
    }
    vector<Point> points;
    for (const auto& l : lines) {
        points.push_back(Point(l[0], l[1]));
        points.push_back(Point(l[2], l[3]));
    }
    fitLine(points, line, DIST_L2, 0, 0.01, 0.01);
    float x1 = 0;
    float y1 = line[3] + line[1]/line[0] * (x1 - line[2]);
    float x2 = roi.cols - 1;
    float y2 = line[3] + line[1]/line[0] * (x2 - line[2]);
    bool isLineValid = (y1 >= 0 && y1 < roi.rows) && (y2 >= 0 && y2 < roi.rows);
    cout << "ROI高度: " << roi.rows << endl;
    return isLineValid;
}

// 相机自检
bool cameraSelfCheck(const Mat& image, const CameraCalibration& calibration, double& deviation) {
    vector<Point2f> corners;
    if (!detectTarget(image, corners)) {
        return false;
    }
    Point2d currentCenter = calculateTargetCenter(corners);
    if (currentCenter.x < 0 || currentCenter.y < 0) {
        return false;
    }
    deviation = norm(currentCenter - calibration.target_center);
    return deviation <= calibration.max_deviation;
}