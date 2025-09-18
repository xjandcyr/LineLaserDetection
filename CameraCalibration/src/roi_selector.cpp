#include "roi_selector.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

// 全局变量用于鼠标回调
static vector<Point2f> g_points;
static Mat g_image;
static string g_windowName;
static bool g_selectionComplete = false;

// 鼠标回调函数
static void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (g_selectionComplete) return;
    
    if (event == EVENT_LBUTTONDOWN) {
        if (g_points.size() < 4) {
            g_points.push_back(Point2f(x, y));
            cout << "添加点 " << g_points.size() << ": (" << x << ", " << y << ")" << endl;
            
            // 在图像上绘制点
            Mat displayImage = g_image.clone();
            for (size_t i = 0; i < g_points.size(); ++i) {
                circle(displayImage, g_points[i], 5, Scalar(0, 255, 0), -1);
                putText(displayImage, to_string(i + 1), g_points[i] + Point2f(10, -10), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }
            
            // 绘制已连接的线段
            for (size_t i = 1; i < g_points.size(); ++i) {
                line(displayImage, g_points[i-1], g_points[i], Scalar(255, 0, 0), 2);
            }
            
            // 如果已经有4个点，连接最后一个点和第一个点
            if (g_points.size() == 4) {
                line(displayImage, g_points[3], g_points[0], Scalar(255, 0, 0), 2);
                g_selectionComplete = true;
                cout << "四边形选择完成！按任意键确认。" << endl;
            }
            
            imshow(g_windowName, displayImage);
        }
    }
    else if (event == EVENT_MOUSEMOVE) {
        // 实时显示鼠标位置和预览线
        if (g_points.size() > 0 && g_points.size() < 4) {
            Mat displayImage = g_image.clone();
            
            // 绘制已确定的点
            for (size_t i = 0; i < g_points.size(); ++i) {
                circle(displayImage, g_points[i], 5, Scalar(0, 255, 0), -1);
                putText(displayImage, to_string(i + 1), g_points[i] + Point2f(10, -10), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }
            
            // 绘制已连接的线段
            for (size_t i = 1; i < g_points.size(); ++i) {
                line(displayImage, g_points[i-1], g_points[i], Scalar(255, 0, 0), 2);
            }
            
            // 绘制预览线
            line(displayImage, g_points.back(), Point2f(x, y), Scalar(0, 255, 255), 2);
            
            imshow(g_windowName, displayImage);
        }
    }
}

// 矩形ROI选择（原有功能）
Rect selectROIFromImage(const Mat& image, const string& windowName) {
    Rect roi = selectROI(windowName, image, false, false);
    destroyWindow(windowName);
    return roi;
}

// 四边形ROI选择（新功能）
QuadROI selectQuadROIFromImage(const Mat& image, const string& windowName) {
    // 初始化全局变量
    g_points.clear();
    g_image = image.clone();
    g_windowName = windowName;
    g_selectionComplete = false;
    
    // 创建窗口并设置鼠标回调，窗口大小设置为800x600
    namedWindow(windowName, WINDOW_NORMAL);     // 修改为可调整大小的窗口类型
    resizeWindow(windowName, WINDOW_WIDTH, WINDOW_HEIGHT);         // 添加窗口尺寸设置
    setMouseCallback(windowName, onMouse);
    
    // 显示图像和说明
    Mat displayImage = image.clone();
    putText(displayImage, "Click 4 points to define the quadrilateral ROI", Point(10, 30), 
           FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
    putText(displayImage, "Press ESC to cancel selection", Point(10, 60), 
           FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
    imshow(windowName, displayImage);
    
    // 等待用户完成选择
    while (!g_selectionComplete) {
        int key = waitKey(1) & 0xFF;
        if (key == 27) { // ESC键
            g_points.clear();
            break;
        }
    }
    
    // 创建QuadROI对象
    QuadROI quadRoi;
    if (g_points.size() == 4) {
        quadRoi = QuadROI(g_points);
    }
    
    destroyWindow(windowName);
    return quadRoi;
}

// 保存矩形ROI到文件
bool saveROIToFile(const Rect& roi, const string& filePath) {
    ofstream file(filePath);
    if (!file.is_open()) return false;
    file << roi.x << " " << roi.y << " " << roi.width << " " << roi.height << endl;
    file.close();
    return true;
}

// 保存四边形ROI到文件
bool saveQuadROIToFile(const QuadROI& quadRoi, const string& filePath) {
    ofstream file(filePath);
    if (!file.is_open()) return false;
    
    file << "QUAD_ROI" << endl; // 文件标识
    // 修改输出格式为带坐标标签的样式
    for (int i = 0; i < 4; ++i) {
        file << "X" << (i+1) << ": " << quadRoi.points[i].x 
             << ", " << quadRoi.points[i].y << endl;
    }
    file.close();
    return true;
}

// 从文件加载矩形ROI
bool loadROIFromFile(Rect& roi, const string& filePath) {
    ifstream file(filePath);
    if (!file.is_open()) return false;
    
    string firstLine;
    getline(file, firstLine);
    
    // 检查是否是四边形ROI文件
    if (firstLine == "QUAD_ROI") {
        file.close();
        return false; // 这是四边形ROI文件，不是矩形ROI
    }
    
    // 重置文件指针到开始
    file.clear();
    file.seekg(0);
    
    int x, y, w, h;
    file >> x >> y >> w >> h;
    if (file.fail()) return false;
    roi = Rect(x, y, w, h);
    return true;
}

// 从文件加载四边形ROI
bool loadQuadROIFromFile(QuadROI& quadRoi, const string& filePath) {
    ifstream file(filePath);
    if (!file.is_open()) return false;
    
    string firstLine;
    getline(file, firstLine);
    
    if (firstLine != "QUAD_ROI") {
        file.close();
        return false; // 不是四边形ROI文件
    }
    
    vector<Point2f> points;
    for (int i = 0; i < 4; ++i) {
        float x, y;
        file >> x >> y;
        if (file.fail()) return false;
        points.push_back(Point2f(x, y));
    }
    
    quadRoi = QuadROI(points);
    return true;
}

// 绘制四边形ROI
void drawQuadROI(Mat& image, const QuadROI& quadRoi, const Scalar& color, int thickness) {
    if (!quadRoi.isValid()) return;
    
    // 绘制四个顶点
    for (size_t i = 0; i < quadRoi.points.size(); ++i) {
        circle(image, quadRoi.points[i], 5, color, -1);
        putText(image, to_string(i + 1), quadRoi.points[i] + Point2f(10, -10), 
               FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    
    // 绘制四边形边线
    for (size_t i = 0; i < quadRoi.points.size(); ++i) {
        Point2f p1 = quadRoi.points[i];
        Point2f p2 = quadRoi.points[(i + 1) % quadRoi.points.size()];
        line(image, p1, p2, color, thickness);
    }
}

// 检查点是否在四边形内（使用射线法）
bool isPointInQuadROI(const Point2f& point, const QuadROI& quadRoi) {
    if (!quadRoi.isValid()) return false;
    
    int intersections = 0;
    const vector<Point2f>& points = quadRoi.points;
    
    for (size_t i = 0; i < points.size(); ++i) {
        Point2f p1 = points[i];
        Point2f p2 = points[(i + 1) % points.size()];
        
        // 检查射线是否与边相交
        if (((p1.y > point.y) != (p2.y > point.y)) &&
            (point.x < (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x)) {
            intersections++;
        }
    }
    
    return (intersections % 2) == 1;
}