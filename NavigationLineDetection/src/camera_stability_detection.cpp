#include "lidar_line_detection.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

using namespace cv;
using namespace std;

// 相机自检相关功能实现
namespace CameraStabilityDetection {

    static std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt("camera_logger", "./navLine_log/camera_stability_detection.log");

    // 标靶配置文件 读取
    DetectionResultCode loadTargetConfig(const string &configPath, LidarLineDetector::TargetConfig &config)
    {
        logger->info("Start reading camera config file: {}", configPath);
        ifstream file(configPath);
        if (!file.is_open())
        {
            logger->error("Unable to open camera config file: {}", configPath);
            return DetectionResultCode::CONFIG_LOAD_FAILED;
        }

        string line;
        bool centerRead = false, toleranceRead = false;
        while (getline(file, line))
        {
            if (line.find("center_x:") == 0)
            {
                if (sscanf(line.c_str(), "center_x: %f", &config.expected_center.x) == 1)
                {
                    centerRead = true;
                }
                else
                {
                    logger->error("Failed to parse center_x: {}", line);
                }
            }
            else if (line.find("center_y:") == 0)
            {
                if (sscanf(line.c_str(), "center_y: %f", &config.expected_center.y) == 1)
                {
                    centerRead = true;
                }
                else
                {
                    logger->error("Failed to parse center_y: {}", line);
                }
            }
            else if (line.find("tolerance:") == 0)
            {
                if (sscanf(line.c_str(), "tolerance: %f", &config.tolerance) == 1)
                {
                    toleranceRead = true;
                }
                else
                {
                    logger->error("Failed to parse tolerance: {}", line);
                }
            }
        }
        file.close();

        if (!centerRead || !toleranceRead)
        {
            logger->error("The config file format is incorrect and the complete information was not read");
            return DetectionResultCode::CONFIG_LOAD_FAILED;
        }
        
        if (centerRead && toleranceRead) {
            logger->info("Camera config read successfully: center=({:.2f}, {:.2f}), tolerance={:.2f}", 
                        config.expected_center.x, config.expected_center.y, config.tolerance);
            return DetectionResultCode::SUCCESS;
        } else {
            return DetectionResultCode::CONFIG_LOAD_FAILED;
        }
    }

    // 检测标靶四个角落的黑色方块
    bool detectTarget(const Mat& image, vector<Point2f>& corners, Mat& displayImage) {
        logger->info("black squares detect start");
        Mat gray, binary;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 80, 255, THRESH_BINARY_INV);

        // 形态学操作去噪
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        vector<Point2f> centers;
        for (const auto& contour : contours) {
            double area = contourArea(contour);

            // 4个黑色方块的面积都为8000左右
            if (area < 5000 || area > 10000)
            {
                continue;
            }

            vector<Point> approx;
            approxPolyDP(contour, approx, arcLength(contour, true) * 0.02, true);
            if (approx.size() == 4 && isContourConvex(approx)) {
                Rect rect = boundingRect(approx);
                double aspect = (double)rect.width / rect.height;
                if (aspect > 0.7 && aspect < 1.3) {
                    Moments m = moments(contour);
                    if (m.m00 != 0) {
                        Point2f center(m.m10/m.m00, m.m01/m.m00);
                        centers.push_back(center);
                        // 在显示图像上绘制检测到的方块
                        circle(displayImage, center, 8, Scalar(0, 255, 0), 2);
                        rectangle(displayImage, rect, Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        // 显示所有的contours
        drawContours(displayImage, contours, -1, Scalar(255, 0, 0), 1.5);
        
        if (centers.size() != 4) {
            logger->error("Failed to detect 4 black squares, found: {} squares", centers.size());
            cv::putText(displayImage, "Target Detection Failed: " + std::to_string(centers.size()) + " targets found", 
                       cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
            return false;
        }
        
        // 按位置排序：左上、右上、左下、右下
        sort(centers.begin(), centers.end(), [](const Point2f& a, const Point2f& b) {
            return a.y < b.y || (a.y == b.y && a.x < b.x);
        });
        if (centers[0].x > centers[1].x) swap(centers[0], centers[1]);
        if (centers[2].x > centers[3].x) swap(centers[2], centers[3]);
        
        corners = centers;
        logger->info("black squares detect end, found 4 black squares");
        return true;
    }

    // 计算标靶中心点
    Point2f calculateTargetCenter(const vector<Point2f>& corners) {
        if (corners.size() != 4) return Point2f(-1, -1);
        float centerX = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4.0f;
        float centerY = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4.0f;
        return Point2f(centerX, centerY);
    }

    // 标靶中心点检测
    DetectionResultCode detectTargetCenter(const Mat &image, Point2f &outCenter, Mat &displayImage)
    {
        logger->info("target surface center detect start");
        displayImage = image.clone();
        
        vector<Point2f> corners;
        if (!detectTarget(image, corners, displayImage)) {
            return DetectionResultCode::CAMERA_SELF_CHECK_FAILED;
        }
        
        outCenter = calculateTargetCenter(corners);
        if (outCenter.x < 0 || outCenter.y < 0) {
            logger->error("Failed to calculate target center, center coordinates: ({:.2f}, {:.2f})", outCenter.x, outCenter.y);
            cv::putText(displayImage, "Center Calculation Failed", 
                       cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
            return DetectionResultCode::CAMERA_SELF_CHECK_FAILED;
        }
        
        circle(displayImage, outCenter, 10, Scalar(0, 255, 0), -1);
        
        logger->info("target surface center detect end, center coordinates: ({:.2f}, {:.2f})", outCenter.x, outCenter.y);
        return DetectionResultCode::SUCCESS;
    }

    // 相机自检函数
    TargetMovementResult_C checkCameraMovement(const Mat &image, const LidarLineDetector::TargetConfig &config, Mat &displayImage)
    {
        logger->info("camera movement detect start");
        // 修复：显式转换枚举类型
        TargetMovementResult_C result{0, 0, 0, 0, static_cast<int>(DetectionResultCode::SUCCESS), ""};
        Point2f currentCenter;
        DetectionResultCode detect_result = detectTargetCenter(image, currentCenter, displayImage);
        if (detect_result != DetectionResultCode::SUCCESS)
        {
            result.error_code = static_cast<int>(detect_result);
            logger->error("Target surface detection failed, error code: {}", result.error_code);
            // 在显示图像上标注失败原因
            cv::putText(displayImage, "Target surface Detection Failed", cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
            return result;
        }

        float dx = currentCenter.x - config.expected_center.x;
        float dy = currentCenter.y - config.expected_center.y;
        result.dx = dx;
        result.dy = dy;
        result.distance = sqrt(dx * dx + dy * dy);
        result.is_stable = (result.distance <= config.tolerance);

        snprintf(result.message, sizeof(result.message),
                 result.is_stable ? "camera is stable, distance: %.1fpx" : "camera is moved, distance: %.1fpx (>%.1fpx)",
                 result.distance, result.distance, config.tolerance);

        // 在显示图像上绘制检测结果
        circle(displayImage, config.expected_center, (int)config.tolerance, Scalar(0, 0, 255), 2);
        line(displayImage, config.expected_center, currentCenter, Scalar(0, 255, 255), 2);
        putText(displayImage, result.message, Point(20, 30), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 2);

        logger->info("Camera self-test completed: {} (distance: {:.1f}px)", 
                    result.is_stable ? "stable" : "moved", result.distance);
        return result;
    }

    // 新接口：只传图片和配置文件路径，图片保存直接在函数内部进行
    TargetMovementResult_C checkCameraMovement(const Mat &image, const std::string &configPath, const std::string &outputDir)
    {
        logger->info("\n===============================================================================");
        LidarLineDetector::TargetConfig config;
        DetectionResultCode err = loadTargetConfig(configPath, config);
        if (err != DetectionResultCode::SUCCESS) {
            TargetMovementResult_C result{};
            result.error_code = static_cast<int>(err);
            snprintf(result.message, sizeof(result.message), "Failed to load target config, error code: %d", result.error_code);
            logger->error("Failed to load target config, error code: {}", result.error_code);
            return result;
        }
        
        // 调用内部函数进行检测
        cv::Mat displayImage;
        TargetMovementResult_C result = checkCameraMovement(image, config, displayImage);
        
        // 如果指定了输出目录，保存图片
        if (!outputDir.empty() && !displayImage.empty()) {
            // 创建输出目录（如果不存在）
            _mkdir(outputDir.c_str());
            
            // 生成带时间戳的文件名
            time_t now = time(0);
            char timeStr[26];
            ctime_s(timeStr, sizeof(timeStr), &now);
            for (int i = 0; timeStr[i]; i++) {
                if (timeStr[i] == ' ' || timeStr[i] == ':' || timeStr[i] == '\n') {
                    timeStr[i] = '_';
                }
            }
            
            std::string fileName = outputDir + "/camera_check_" + timeStr + ".jpg";
            if (cv::imwrite(fileName, displayImage)) {
                logger->info("Camera self-test image saved: {}", fileName);
                // 将保存的文件路径添加到结果消息中
                std::string originalMessage = result.message;
                snprintf(result.message, sizeof(result.message), "%s | image: %s", 
                        originalMessage.c_str(), fileName.c_str());
            } else {
                logger->error("Failed to save camera self-test image: {}", fileName);
            }
        }
        
        return result;
    }

} // namespace CameraStabilityDetection

// 封装类实现 - 相机自检相关方法
DetectionResultCode CLidarLineDetector::loadTargetConfig(const char *configPath, LidarLineDetector::TargetConfig &config)
{
    return CameraStabilityDetection::loadTargetConfig(configPath, config);
}

TargetMovementResult_C CLidarLineDetector::checkCameraStability(const TCMat_C image, const char* configPath)
{
    Mat image_cpp(image.rows, image.cols, image.type, image.data);
    return CameraStabilityDetection::checkCameraMovement(image_cpp, std::string(configPath), m_outputDir);
}

// C 接口实现 - 相机自检相关
extern "C"
{
    Smpclass_API DetectionResultCode CLidarLineDetector_loadTargetConfig(CLidarLineDetector *instance, const char *configPath, TTargetConfig_C *config)
    {
        LidarLineDetector::TargetConfig internalConfig;
        auto err = instance->loadTargetConfig(configPath, internalConfig);
        if (err == DetectionResultCode::SUCCESS)
        {
            config->center_x = internalConfig.expected_center.x;
            config->center_y = internalConfig.expected_center.y;
            config->tolerance = internalConfig.tolerance;
        }
        return err;
    }

    Smpclass_API TargetMovementResult_C CLidarLineDetector_checkCameraStability(CLidarLineDetector *instance, const TCMat_C image, const char* configPath)
    {
        return instance->checkCameraStability(image, configPath);
    }
}