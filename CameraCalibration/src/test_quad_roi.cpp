#include "roi_selector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

int main() {
    // 设置控制台为UTF-8编码
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    cout << "四边形ROI测试程序" << endl;
    cout << "==================" << endl;
    
    // 测试1: 加载已保存的四边形ROI
    cout << "测试1: 加载已保存的四边形ROI..." << endl;
    QuadROI loadedQuadRoi;
    if (loadQuadROIFromFile(loadedQuadRoi, "quad_roi.txt")) {
        cout << "成功加载四边形ROI！" << endl;
        cout << "顶点坐标：" << endl;
        for (size_t i = 0; i < loadedQuadRoi.points.size(); ++i) {
            cout << "点" << (i+1) << ": (" << loadedQuadRoi.points[i].x 
                 << ", " << loadedQuadRoi.points[i].y << ")" << endl;
        }
        
        // 创建一个测试图像来显示加载的ROI
        Mat testImage(600, 800, CV_8UC3, Scalar(255, 255, 255));
        
        // 绘制四边形ROI
        drawQuadROI(testImage, loadedQuadRoi, Scalar(0, 255, 0), 3);
        
        // 添加说明文字
        putText(testImage, "加载的四边形ROI", Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 2);
        putText(testImage, "点击测试点检测功能", Point(10, 60), 
               FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 1);

        // 设置鼠标回调进行点检测测试
        namedWindow("加载的四边形ROI", WINDOW_AUTOSIZE);
        setMouseCallback("加载的四边形ROI", [](int event, int x, int y, int flags, void* userdata) {
            if (event == EVENT_LBUTTONDOWN) {
                Point2f testPoint(x, y);
                bool inside = isPointInQuadROI(testPoint, *(QuadROI*)userdata);
                
                Mat testImage(600, 800, CV_8UC3, Scalar(255, 255, 255));
                drawQuadROI(testImage, *(QuadROI*)userdata, Scalar(0, 255, 0), 3);
                
                // 绘制测试点
                circle(testImage, testPoint, 8, inside ? Scalar(0, 0, 255) : Scalar(255, 0, 0), -1);
                
                // 显示结果文字
                string resultText = inside ? "在四边形内" : "在四边形外";
                Scalar textColor = inside ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
                putText(testImage, resultText, Point(x+10, y-10), 
                       FONT_HERSHEY_SIMPLEX, 1.2, textColor, 2);
                
                putText(testImage, "加载的四边形ROI", Point(10, 30), 
                       FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 2);
                putText(testImage, "点击测试点检测功能", Point(10, 60), 
                       FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 1);
                
                imshow("加载的四边形ROI", testImage);
            }
        }, &loadedQuadRoi);

        imshow("加载的四边形ROI", testImage);
        cout << "按任意键继续..." << endl;
        waitKey(0);
    } else {
        cout << "未找到已保存的四边形ROI文件，请先运行主程序创建四边形ROI" << endl;
    }
    
    // 测试2: 创建示例四边形ROI
    cout << "测试2: 创建示例四边形ROI..." << endl;
    vector<Point2f> samplePoints = {
        Point2f(100, 100),
        Point2f(300, 80),
        Point2f(350, 250),
        Point2f(150, 300)
    };
    
    QuadROI sampleQuadRoi(samplePoints);
    
    // 创建测试图像
    Mat sampleImage(400, 500, CV_8UC3, Scalar(255, 255, 255));
    drawQuadROI(sampleImage, sampleQuadRoi, Scalar(255, 0, 0), 2);
    
    putText(sampleImage, "示例四边形ROI", Point(10, 30), 
           FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 2);
    putText(sampleImage, "点击测试点检测功能", Point(10, 60), 
           FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 1);

    // 设置鼠标回调
    namedWindow("示例四边形ROI", WINDOW_AUTOSIZE);
    setMouseCallback("示例四边形ROI", [](int event, int x, int y, int flags, void* userdata) {
        if (event == EVENT_LBUTTONDOWN) {
            Point2f testPoint(x, y);
            bool inside = isPointInQuadROI(testPoint, *(QuadROI*)userdata);
            
            Mat testImage(400, 500, CV_8UC3, Scalar(255, 255, 255));
            drawQuadROI(testImage, *(QuadROI*)userdata, Scalar(255, 0, 0), 2);
            
            // 绘制测试点
            circle(testImage, testPoint, 8, inside ? Scalar(0, 0, 255) : Scalar(255, 0, 0), -1);
            
            // 显示结果文字
            string resultText = inside ? "在四边形内" : "在四边形外";
            Scalar textColor = inside ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
            putText(testImage, resultText, Point(x+10, y-10), 
                   FONT_HERSHEY_SIMPLEX, 1.2, textColor, 2);
            
            putText(testImage, "示例四边形ROI", Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 2);
            putText(testImage, "点击测试点检测功能", Point(10, 60), 
                   FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 0), 1);
            
            imshow("示例四边形ROI", testImage);
        }
    }, &sampleQuadRoi);
    
    imshow("示例四边形ROI", sampleImage);
    
    // 保存示例四边形ROI
    if (saveQuadROIToFile(sampleQuadRoi, "NavLineCheckRoi", "sample_quad_roi.txt")) {
        cout << "示例四边形ROI已保存到 sample_quad_roi.txt" << endl;
    } else {
        cout << "保存示例四边形ROI失败！" << endl;
    }

    cout << "按ESC键退出..." << endl;
    while (true) {
        int key = waitKey(1) & 0xFF;
        if (key == 27) break; // ESC键
    }
    
    return 0;
}