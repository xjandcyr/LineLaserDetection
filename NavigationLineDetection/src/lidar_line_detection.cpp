#include "lidar_line_detection.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdint>
#include <cmath>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <direct.h> // Windows下创建文件夹

using namespace cv;
using namespace std;

// 将所有LidarLineDetector相关函数实现放入命名空间
namespace LidarLineDetector {

    // 版本信息
    static const char *versionString = "1.0.7";

    static std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt("lidar_logger", "./navLine_log/lidar_line_detection.log");

    VersionInfo getVersionInfo()
    {
        return {
            LIDAR_LINE_DETECTION_VERSION_MAJOR,
            LIDAR_LINE_DETECTION_VERSION_MINOR,
            LIDAR_LINE_DETECTION_VERSION_PATCH,
            versionString};
    }

    const char *getVersionString()
    {
        return versionString;
    }

    int getVersionMajor()
    {
        return LIDAR_LINE_DETECTION_VERSION_MAJOR;
    }

    int getVersionMinor()
    {
        return LIDAR_LINE_DETECTION_VERSION_MINOR;
    }

    int getVersionPatch()
    {
        return LIDAR_LINE_DETECTION_VERSION_PATCH;
    }

    // 读取四边形ROI配置文件
    DetectionResultCode readQuadROIFromConfig(const string &configPath, QuadROI &quadRoi)
    {
        logger->info("ROI configuration path: {}", configPath);
        FileStorage fs(configPath, cv::FileStorage::READ);

        // 打开YAML配置文件
        if (!fs.isOpened()) {
            logger->error("Failed to open config file: {}", configPath);
            return DetectionResultCode::CONFIG_LOAD_FAILED;
        }

        // 读取NavLineCheckRoi节点
        cv::FileNode roiNode = fs["NavLineCheckRoi"];
        if (roiNode.empty()) {
            logger->error("NavLineCheckRoi node not found or empty in config file: {}", configPath);
            fs.release();
            return DetectionResultCode::CONFIG_LOAD_FAILED;
        }

        // 检查是否是序列且包含恰好4个点
        if (roiNode.type() != cv::FileNode::SEQ || roiNode.size() != 4) {
            logger->error("NavLineCheckRoi must be a sequence with exactly 4 points");
            fs.release();
            return DetectionResultCode::CONFIG_LOAD_FAILED;
        }

        // 遍历序列中的四个点
        for (int i = 0; i < 4; ++i) {
            cv::FileNode pointNode = roiNode[i];
            // 检查点节点是否包含x和y字段
            if (pointNode["x"].empty() || pointNode["y"].empty()) {
                logger->error("Point at index {} is missing x or y coordinate", i);
                fs.release();
            }
            
            // 读取x和y坐标值
            int x = (int)pointNode["x"];
            int y = (int)pointNode["y"];
            // 赋值给QuadROI结构
            quadRoi.points[i] = cv::Point(x, y);
            logger->info("Loaded point {}: ({}, {})", i, x, y);
        }

        fs.release();
        logger->info("Successfully loaded ROI configuration");

        // 验证四边形是否有效（简单的面积检查）
        double area = 0;
        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            area += quadRoi.points[i].x * quadRoi.points[j].y;
            area -= quadRoi.points[j].x * quadRoi.points[i].y;
        }
        area = abs(area) / 2.0;
        if (area < 100) // 最小面积阈值
        {
            logger->error("The quadrilateral ROI area is too small: {}", area);
            return DetectionResultCode::ROI_INVALID;
        }

        logger->info("Quad ROI config read successfully: P1({},{}), P2({},{}), P3({},{}), P4({},{})", 
                    quadRoi.points[0].x, quadRoi.points[0].y,
                    quadRoi.points[1].x, quadRoi.points[1].y,
                    quadRoi.points[2].x, quadRoi.points[2].y,
                    quadRoi.points[3].x, quadRoi.points[3].y);
        return DetectionResultCode::SUCCESS;
    }

    // 生成带时间和SN的文件名
    string generateFileName(const string &basePath, const string &sn)
    {
        time_t now = time(0);
        char timeStr[26];
        ctime_s(timeStr, sizeof(timeStr), &now);
        for (int i = 0; timeStr[i]; i++)
            if (timeStr[i] == ' ' || timeStr[i] == ':' || timeStr[i] == '\n')
                timeStr[i] = '_';
        return basePath + "_" + sn + "_" + timeStr + ".jpg";
    }

    // 保存失败图像的函数
    void saveResultImage(const cv::Mat& sourceImage,
                        const std::string& message,
                        const std::string& sn,
                        const QuadROI& quadRoi,
                        const std::string& outputDir,
                        LidarDetectionResult &result)
    {
        if (! outputDir.empty())
        {
            cv::Mat resultImage = sourceImage.clone();
            for(int i = 0; i < 4; i++)
            {
                cv::line(resultImage, quadRoi.points[i], quadRoi.points[(i + 1) % 4], cv::Scalar(0, 255, 0), 1);
            }
            cv::putText(resultImage, message, cv::Point(20, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
            
            std::string fileName = generateFileName(outputDir + "/result", sn);
            if (cv::imwrite(fileName, resultImage))
            {
                logger->info("Result image saved: {}", fileName);
                result.image_path = fileName;
            }
            else
            {
                logger->error("Failed to save result image: {}", fileName);
                result.status = DetectionResultCode::IMAGE_SAVE_FAILED;
            }
        }
        else
        {
            logger->error("Output directory is empty");
            result.status = DetectionResultCode::IMAGE_SAVE_FAILED;
        }
    }

    // 使用四边形ROI的激光线检测核心函数
    LidarDetectionResult detectLidarLineWithQuadROI(const cv::Mat& image, const QuadROI& quadRoi, const std::string& sn, const std::string& outputDir)
    {
        logger->info("Start laser line detection within the ROI");
        LidarDetectionResult result;
        result.status = DetectionResultCode::NOT_FOUND;
        result.line_angle = 0.0f;
        result.image_path = "";

        // 检查四边形ROI是否在图像范围内
        for(int i = 0; i < 4; i++)
        {
            if(quadRoi.points[i].x < 0 || quadRoi.points[i].y < 0 ||
               quadRoi.points[i].x >= image.cols || quadRoi.points[i].y >= image.rows)
            {
                logger->error("ROI exceeds the image range, point{}: ({}, {})",i, quadRoi.points[i].x, quadRoi.points[i].y);
                string failMsg = "ROI exceeds the image range, point" + std::to_string(quadRoi.points[i].x) + ", " + std::to_string(quadRoi.points[i].y) + ")";
                saveResultImage(image, failMsg, sn, quadRoi, outputDir, result);
                result.status = DetectionResultCode::ROI_INVALID;
                return result;
            }
        }

        // 创建四边形ROI的掩码
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        std::vector<cv::Point> quadPoints;
        for (int i = 0; i < 4; i++)
        {
            quadPoints.push_back(quadRoi.points[i]);
        }
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{quadPoints}, cv::Scalar(255));

        // 提取ROI区域
        cv::Mat roiMat;
        image.copyTo(roiMat, mask);
        if (roiMat.empty())
        {
            logger->error("Failed to extract the quadrilateral ROI area, ROI is empty");
            result.status = DetectionResultCode::ROI_INVALID;
            return result;
        }

        // 灰度化
        cv::Mat gray;
        cv::cvtColor(roiMat, gray, cv::COLOR_BGR2GRAY);

        // 提取所有高亮点
        std::vector<cv::Point> laserPoints;
        for (int y = 0; y < gray.rows; ++y) {
            for (int x = 0; x < gray.cols; ++x) {
                if (gray.at<uchar>(y, x) > 220) { // 阈值可调
                    laserPoints.emplace_back(x, y);
                }
            }
        }

        // 可视化激光点
        cv::Mat debugPoints = roiMat.clone();
        for (const auto& pt : laserPoints) {
            cv::circle(debugPoints, pt, 1, cv::Scalar(0, 0, 255), -1);
        }
        if (!outputDir.empty()) {
            std::string debugFileName = generateFileName(outputDir + "/ROI_laser_points", sn);
            cv::imwrite(debugFileName, debugPoints);
        }

        // 判据1：导航激光线所占像素数量过小，认为激光线不在ROI区域内
        if (laserPoints.size() < 100) {
            logger->error("The laser line is not within the ROI area, number of points: {}", laserPoints.size());
            string failMsg = "number of Laser Points: " + std::to_string(laserPoints.size());
            saveResultImage(image, failMsg, sn, quadRoi, outputDir, result);
            result.status = DetectionResultCode::OUT_OF_ROI;
            return result;
        }

        // 用fitLine拟合直线
        cv::Vec4f line;
        cv::fitLine(laserPoints, line, cv::DIST_L2, 0, 0.01, 0.01);
        float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];

        // 判据2：RMS误差
        double sumDist2 = 0;
        for (const auto& pt : laserPoints) {
            double dist = std::abs(vy * (pt.x - x0) - vx * (pt.y - y0)) / std::sqrt(vx * vx + vy * vy);
            sumDist2 += dist * dist;
        }
        double rms = std::sqrt(sumDist2 / laserPoints.size());
        logger->info("Navigation laser line RMS: {}", rms);
        if (rms > 5.0) {
            logger->error("RMS is too high, RMS: {}", rms);
            string failMsg = "RMS is too high, RMS: " + std::to_string(rms);
            saveResultImage(image, failMsg, sn, quadRoi, outputDir, result);
            result.status = DetectionResultCode::NOT_FOUND;
            return result;
        }

        // 判据3：投影长度
        std::vector<double> projections;
        for (const auto& pt : laserPoints) {
            double proj = (pt.x - x0) * vx + (pt.y - y0) * vy;
            projections.push_back(proj);
        }
        auto minmax = std::minmax_element(projections.begin(), projections.end());
        double length = *minmax.second - *minmax.first;

        // 计算四边形ROI的最长的一条边作为长度参考
        double roiWidth = 0;
        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            double dist = cv::norm(quadRoi.points[i] - quadRoi.points[j]);
            roiWidth = std::max(roiWidth, dist);
        }
        logger->info("Navigation laser line length: {}, ROI Width: {}", length, roiWidth);

        // 阈值可根据实际调整
        if (length < roiWidth * 0.97)
        {
            logger->info("Navigation laser line length: {}, ROI Width: {}", length, roiWidth);
            string failMsg = "The length is too short, length: " + std::to_string(length);
            saveResultImage(image, failMsg, sn, quadRoi, outputDir, result);
            result.status = DetectionResultCode::NOT_FOUND;
            return result;
        }

        float lineAngle = std::atan2(line[1], line[0]);     // atan2(y, x)
        result.line_angle = lineAngle;

        // 如果输出目录不为空，保存结果图像
        if (!outputDir.empty())
        {
            cv::Mat resultImage = image.clone();
            // 绘制四边形ROI
            for (int j = 0; j < 4; j++)
            {
                cv::line(resultImage, quadRoi.points[j], quadRoi.points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            
            // 画直线段（只覆盖所有高亮点）
            std::vector<double> projections;
            for (const auto& pt : laserPoints) {
                double proj = (pt.x - x0) * vx + (pt.y - y0) * vy;
                projections.push_back(proj);
            }
            auto minmax = std::minmax_element(projections.begin(), projections.end());
            double minProj = *minmax.first;
            double maxProj = *minmax.second;
            // 计算直线段的两个端点（ROI内坐标）
            cv::Point pt1_roi(x0 + minProj * vx, y0 + minProj * vy);
            cv::Point pt2_roi(x0 + maxProj * vx, y0 + maxProj * vy);
            // 获取四边形ROI四个顶点y坐标
            int x1_y = quadRoi.points[0].y;
            int x2_y = quadRoi.points[1].y;
            int x3_y = quadRoi.points[2].y;
            int x4_y = quadRoi.points[3].y;
            // 判断left.y在X1~X4，right.y在X2~X3
            bool left_in = (pt1_roi.y >= x1_y - 5) && (pt1_roi.y <= x4_y + 5);
            bool right_in = (pt2_roi.y >= x2_y -5) && (pt2_roi.y <= x3_y + 5);
            if (!(left_in && right_in)) {
                logger->error("The laser line endpoint exceeds the ROI boundary: left_y={}, right_y={}", pt1_roi.y, pt2_roi.y);
                result.status = DetectionResultCode::OUT_OF_ROI;
                return result;
            }
            cv::line(resultImage, pt1_roi, pt2_roi, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

            std::string fileName = generateFileName(outputDir + "/result", sn);
            if (cv::imwrite(fileName, resultImage))
            {
                logger->info("Image saved successfully: {}", fileName);
                result.image_path = fileName;
            }
            else
            {
                logger->error("Image saved failed: {}", fileName);
            }
        }
        result.status = DetectionResultCode::SUCCESS;
        logger->info("Quad ROI laser line detection successful, angle: {:.2f}°, number of points: {}, RMS: {:.2f}, length: {:.2f}", 
                    lineAngle * 180.0 / CV_PI, laserPoints.size(), rms, length);
        return result;
    }

    // 激光线检测主函数
    LidarLineResult detect(const cv::Mat &image, const std::string &configPath, const std::string &sn, const std::string &outputDir)
    {
        logger->info("\n===============================================================================");
        logger->info("Start lidar line detection, sn: {}", sn);
        LidarLineResult result{false, 0, "", DetectionResultCode::SUCCESS};
        
        // 首先尝试读取四边形ROI配置
        QuadROI quadRoi;
        DetectionResultCode quadRoiResult = readQuadROIFromConfig(configPath, quadRoi);
        
        if (quadRoiResult == DetectionResultCode::SUCCESS)
        {
            logger->info("Use quadrilateral ROI for detection");
            LidarDetectionResult detectionResult = detectLidarLineWithQuadROI(image, quadRoi, sn, outputDir);
            
            if (detectionResult.status != DetectionResultCode::SUCCESS)
            {
                logger->error("Quad ROI laser line detection failed, Error Code: {}", static_cast<int>(detectionResult.status));
                result.error_code = detectionResult.status;
                return result;
            }
            result.line_angle = detectionResult.line_angle;
            result.image_path = detectionResult.image_path;
            result.line_detected = true;
            result.error_code = DetectionResultCode::SUCCESS;
            return result;
        }
        else
        {
            logger->info("Quad ROI configuration read failed, Error Code: {}", static_cast<int>(quadRoiResult));
            result.error_code = quadRoiResult;
            return result;
        }
    }
} // namespace LidarLineDetector


// 封装类实现
DetectionResultCode CLidarLineDetector::initialize(const char *configPath)
{
    return LidarLineDetector::readQuadROIFromConfig(configPath, m_quadRoi);
}

void CLidarLineDetector::setROI(int x, int y, int width, int height) { m_roi = {x, y, width, height}; }
void CLidarLineDetector::setSn(const char *sn) { m_sn = sn ? sn : ""; }
void CLidarLineDetector::setOutputDir(const char *outputDir) { m_outputDir = outputDir ? outputDir : ""; }

TLidarLineResult_C CLidarLineDetector::detect(const TCMat_C image, const char* configPath)
{
    Mat image_cpp(image.rows, image.cols, image.type, image.data);
    auto result = LidarLineDetector::detect(image_cpp, configPath, m_sn, m_outputDir);

    TLidarLineResult_C result_c;
    result_c.line_detected = result.line_detected;
    result_c.line_angle = result.line_angle;
    snprintf(result_c.image_path, sizeof(result_c.image_path), "%s", result.image_path.c_str());
    result_c.error_code = static_cast<int>(result.error_code);
    return result_c;
}

// 版本信息实现
VersionInfo CLidarLineDetector::getVersionInfo()
{
    return LidarLineDetector::getVersionInfo();
}

const char *CLidarLineDetector::getVersionString()
{
    return LidarLineDetector::getVersionString();
}

int CLidarLineDetector::getVersionMajor()
{
    return LidarLineDetector::getVersionMajor();
}

int CLidarLineDetector::getVersionMinor()
{
    return LidarLineDetector::getVersionMinor();
}

int CLidarLineDetector::getVersionPatch()
{
    return LidarLineDetector::getVersionPatch();
}

// C 接口实现
extern "C"
{
    Smpclass_API CLidarLineDetector *CLidarLineDetector_new()
    {
        return new CLidarLineDetector();
    }

    Smpclass_API void CLidarLineDetector_delete(CLidarLineDetector *instance)
    {
        delete instance;
    }

    Smpclass_API DetectionResultCode CLidarLineDetector_initialize(CLidarLineDetector *instance, const char *configPath)
    {
        return instance->initialize(configPath);
    }

    Smpclass_API void CLidarLineDetector_setROI(CLidarLineDetector *instance, int x, int y, int width, int height)
    {
        instance->setROI(x, y, width, height);
    }

    Smpclass_API void CLidarLineDetector_setSn(CLidarLineDetector *instance, const char *sn)
    {
        instance->setSn(sn);
    }

    Smpclass_API void CLidarLineDetector_setOutputDir(CLidarLineDetector *instance, const char *outputDir)
    {
        instance->setOutputDir(outputDir);
    }

    Smpclass_API TLidarLineResult_C CLidarLineDetector_detect(CLidarLineDetector *instance, const TCMat_C image, const char* configPath)
    {
        return instance->detect(image, configPath);
    }


    // 版本信息C接口实现
    Smpclass_API VersionInfo LidarLineDetector_GetVersionInfo()
    {
        return LidarLineDetector::getVersionInfo();
    }

    Smpclass_API const char *LidarLineDetector_GetVersionString()
    {
        return LidarLineDetector::getVersionString();
    }

    Smpclass_API int LidarLineDetector_GetVersionMajor()
    {
        return LidarLineDetector::getVersionMajor();
    }

    Smpclass_API int LidarLineDetector_GetVersionMinor()
    {
        return LidarLineDetector::getVersionMinor();
    }

    Smpclass_API int LidarLineDetector_GetVersionPatch()
    {
        return LidarLineDetector::getVersionPatch();
    }
}
