#include "camera_functions.h"
#include <windows.h>
#include <commdlg.h>
#include <iostream>
#include "roi_selector.h"
using namespace std;
using namespace cv;

// 全局变量用于鼠标回调
static Mat g_testImage;
static QuadROI g_testQuadRoi;

// 鼠标回调函数用于点检测测试
static void onMouseTest(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point2f testPoint(x, y);
        bool inside = isPointInQuadROI(testPoint, g_testQuadRoi);
        
        Mat testImage = g_testImage.clone();
        drawQuadROI(testImage, g_testQuadRoi, Scalar(0, 255, 0), 2);
        circle(testImage, testPoint, 8, inside ? Scalar(0, 0, 255) : Scalar(255, 0, 0), -1);
        putText(testImage, inside ? "Yes" : "No", 
               Point(x+10, y-10), FONT_HERSHEY_SIMPLEX, 1.2, 
               inside ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 2);
        // 调整窗口大小
        resizeWindow("Test", WINDOW_WIDTH, WINDOW_HEIGHT);
        imshow("Test", testImage);
    }
}

// 获取用户选择的图片路径
std::string openFileDialog() {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "图片文件\0*.jpg;*.png;*.bmp\0所有文件\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    ofn.lpstrTitle = "选择图片";
    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}

int main() {
    // 设置控制台为UTF-8编码，防止中文输出乱码
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // 弹窗选择图片
    std::string imagePath = openFileDialog();
    if (imagePath.empty()) {
        std::cout << "未选择图片，程序退出。" << std::endl;
        return -1;
    }
    
    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "无法读取图像: " << imagePath << endl;
        return -1;
    }
    
    cout << "请选择ROI类型：" << endl;
    cout << "1. 相机自检ROI" << endl;
    cout << "2. 导航线检测ROI" << endl;
    cout << "3. 相机标定" << endl;
    cout << "请输入选择 (1-3): ";
    
    int choice;
    cin >> choice;
    
    switch (choice) {
        case 1: {
            // 相机自检ROI选择
            cout << "选择相机自检ROI区域" << endl;
            cout << "请点击4个点来定义自检区域，按ESC取消选择" << endl;
            
            QuadROI quadRoi = selectQuadROIFromImage(image, "Select QuadROI");
            if (quadRoi.isValid()) {
                cout << "已选择相机自检ROI区域，顶点坐标：" << endl;
                for (size_t i = 0; i < quadRoi.points.size(); ++i) {
                    cout << "点" << (i+1) << ": (" << quadRoi.points[i].x 
                         << ", " << quadRoi.points[i].y << ")" << endl;
                }
                cout << "外接矩形: x=" << quadRoi.boundingBox.x 
                     << ", y=" << quadRoi.boundingBox.y 
                     << ", w=" << quadRoi.boundingBox.width 
                     << ", h=" << quadRoi.boundingBox.height << endl;
                
                // 绘制四边形ROI
                Mat resultImage = image.clone();
                drawQuadROI(resultImage, quadRoi, Scalar(0, 255, 0), 2);
                imshow("QuadROI Result", resultImage);
                
                // 修改调用处的路径格式
                if (saveQuadROIToFile(quadRoi, "CameraSelfCheckRoi", "../../config/CameraSelfCheckRoi.yml"))
                {
                    cout << "相机自检ROI已保存到 config/CameraSelfCheckRoi.yml" << endl;
                } else {
                    cout << "保存相机自检ROI失败！" << endl;
                }
                
                // 演示点检测功能
                cout << "点击图像上的任意位置测试点是否在四边形内..." << endl;
                namedWindow("Test", WINDOW_NORMAL);                 // 修改为可调整大小的窗口类型
                resizeWindow("Test", WINDOW_WIDTH, WINDOW_HEIGHT);  // 设置窗口尺寸
                
                // 设置全局变量供回调函数使用
                g_testImage = image.clone();
                g_testQuadRoi = quadRoi;
                
                setMouseCallback("Test", onMouseTest);
                
                // 显示测试窗口
                resizeWindow("Test", WINDOW_WIDTH, WINDOW_HEIGHT);
                imshow("Test", resultImage);
                waitKey(0);
            } else {
                cout << "未选择有效四边形ROI。" << endl;
            }
            break;
        }

        case 2: {
            // 导航线检测ROI选择
            cout << "选择导航线检测ROI区域" << endl;
            cout << "请点击4个点来定义检测区域，按ESC取消选择" << endl;
            
            QuadROI quadRoi = selectQuadROIFromImage(image, "Select QuadROI");
            if (quadRoi.isValid()) {
                cout << "已选择导航线检测ROI区域，顶点坐标：" << endl;
                for (size_t i = 0; i < quadRoi.points.size(); ++i) {
                    cout << "点" << (i+1) << ": (" << quadRoi.points[i].x 
                         << ", " << quadRoi.points[i].y << ")" << endl;
                }
                cout << "外接矩形: x=" << quadRoi.boundingBox.x 
                     << ", y=" << quadRoi.boundingBox.y 
                     << ", w=" << quadRoi.boundingBox.width 
                     << ", h=" << quadRoi.boundingBox.height << endl;
                
                // 绘制四边形ROI
                Mat resultImage = image.clone();
                drawQuadROI(resultImage, quadRoi, Scalar(0, 255, 0), 2);
                imshow("QuadROI Result", resultImage);
                
                if (saveQuadROIToFile(quadRoi, "NavLineCheckRoi", "../../config/NavLineCheckRoi.yml")) {
                    cout << "导航线检测ROI已保存到 config/NavLineCheckRoi.yml" << endl;
                } else {
                    cout << "保存导航线检测ROI失败！" << endl;
                }
                
                // 演示点检测功能
                cout << "点击图像上的任意位置测试点是否在四边形内..." << endl;
                namedWindow("Test", WINDOW_NORMAL);                 // 修改为可调整大小的窗口类型
                resizeWindow("Test", WINDOW_WIDTH, WINDOW_HEIGHT);  // 设置窗口尺寸
                
                // 设置全局变量供回调函数使用
                g_testImage = image.clone();
                g_testQuadRoi = quadRoi;
                
                setMouseCallback("Test", onMouseTest);
                
                // 显示测试窗口
                resizeWindow("Test", WINDOW_WIDTH, WINDOW_HEIGHT);
                imshow("Test", resultImage);
                waitKey(0);
            } else {
                cout << "未选择有效四边形ROI。" << endl;
            }
            break;
        }

        case 3: {
            // 相机标定模式
            cout << "进行相机标定..." << endl;
            vector<Point2f> corners;
            if (detectTarget(image, corners)) {
                Point2d center = calculateTargetCenter(corners);
                cout << "标靶中心点: (" << center.x << ", " << center.y << ")" << endl;
                
                // 保存标定数据
                CameraCalibration calibration;
                calibration.target_center = center;
                if (calibration.savePointToFile("../../config/SelfCheckCenterPoint.yml")) {
                    cout << "相机标定成功，数据已保存到 config/SelfCheckCenterPoint.yml" << endl;
                    
                    // 在图像上绘制角点和中心点
                    Mat resultImage = image.clone();
                    for (const auto& corner : corners) {
                        circle(resultImage, corner, 5, Scalar(255, 0, 0), -1); // 蓝色圆点标记角点
                    }
                    // 连接四个角点
                    for (int i = 0; i < 4; ++i) {
                        line(resultImage, corners[i], corners[(i+1)%4], Scalar(0, 255, 0), 2); // 绿色线
                    }
                    circle(resultImage, center, 5, Scalar(0, 255, 255), -1); // 黄色圆点标记中心点
                    line(resultImage, Point(center.x-10, center.y), Point(center.x+10, center.y), Scalar(0, 255, 255), 1);
                    line(resultImage, Point(center.x, center.y-10), Point(center.x, center.y+10), Scalar(0, 255, 255), 1);
                    
                    // 显示结果
                    namedWindow("Camera Calibration Result", WINDOW_NORMAL);
                    resizeWindow("Camera Calibration Result", WINDOW_WIDTH, WINDOW_HEIGHT);
                    imshow("Camera Calibration Result", resultImage);
                    waitKey(0);
                } else {
                    cout << "保存标定数据失败" << endl;
                }
            } else {
                cout << "相机标定失败，无法检测到标靶" << endl;
            }
            break;
        }
        
        default:
            cout << "无效选择！" << endl;
            break;
    }
    
    return 0;
}