#ifndef LIDAR_LINE_DETECTION_H
#define LIDAR_LINE_DETECTION_H

// ============================================================================
// 包含文件
// ============================================================================
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ============================================================================
// 版本信息定义
// ============================================================================
#define LIDAR_LINE_DETECTION_VERSION_MAJOR 1
#define LIDAR_LINE_DETECTION_VERSION_MINOR 0
#define LIDAR_LINE_DETECTION_VERSION_PATCH 6

// ============================================================================
// 导出宏定义
// ============================================================================
#ifdef _WIN32
    #define Smpclass_API __declspec(dllexport)
#else
    #define Smpclass_API
#endif

// ============================================================================
// 错误码和状态定义
// ============================================================================
enum class DetectionResultCode {
    SUCCESS = 0,                    // 检测成功
    NOT_FOUND = 1,                  // 检测不到线
    OUT_OF_ROI = 2,                 // 线不在ROI范围内
    IMAGE_LOAD_FAILED = 3,          // 图像加载失败
    CONFIG_LOAD_FAILED = 4,         // 配置加载失败
    ROI_INVALID = 5,                // ROI无效
    IMAGE_SAVE_FAILED = 6,          // 图像保存失败
    CAMERA_SELF_CHECK_FAILED = 7,   // 相机自检失败
    UNKNOWN_ERROR = 100             // 未知错误
};

// ============================================================================
// 版本信息结构体
// ============================================================================
struct VersionInfo {
    int major;
    int minor;
    int patch;
    const char* versionString;
};

// ============================================================================
// C接口数据结构定义
// ============================================================================
#pragma pack(push, 1)

// OpenCV Mat的C接口结构体
struct TCMat_C {
    int rows;
    int cols;
    int type;
    void* data;
};

// ROI配置的C接口结构体
struct TROIConfig_C {
    int x;
    int y;
    int width;
    int height;
};

// 激光线检测结果的C接口结构体
struct TLidarLineResult_C {
    bool line_detected;
    float line_angle;           // 直线角度（弧度）
    char image_path[256];
    int error_code;
};

// 目标配置的C接口结构体
struct TTargetConfig_C {
    float center_x;
    float center_y;
    float tolerance;            // 允许的像素偏差
};

// 相机移动检测结果的C接口结构体
struct TargetMovementResult_C {
    int is_stable;
    float dx;
    float dy;
    float distance;
    int error_code;
    char message[256];
};

// 激光线检测结果的C接口结构体（简化版）
struct TLidarDetectionResult_C {
    int status;                 // 0:成功 1:未检测到线 2:线不在ROI
    float line_angle;
    char image_path[256];
};

#pragma pack(pop)

// ============================================================================
// C++ 命名空间：激光线检测核心功能
// ============================================================================
namespace LidarLineDetector {

    // ------------------------------------------------------------------------
    // 数据结构定义
    // ------------------------------------------------------------------------
    
    // ROI配置结构体
    struct ROI {
        int x;
        int y;
        int width;
        int height;
    };

    // 四边形ROI结构体
    struct QuadROI {
        cv::Point2i points[4];  // 四个坐标点，按顺时针或逆时针顺序
    };

    // 激光线检测结果结构体
    struct LidarDetectionResult {
        DetectionResultCode status;   // 检测状态/错误码
        float line_angle;             // 检测到的线的角度（仅SUCCESS时有效）
        std::string image_path;       // 结果图像路径
    };

    // 激光线检测结果结构体（简化版）
    struct LidarLineResult {
        bool line_detected;
        float line_angle;
        std::string image_path;
        DetectionResultCode error_code;
    };

    // 目标配置结构体
    struct TargetConfig {
        cv::Point2f expected_center;
        float tolerance;
    };

    // ------------------------------------------------------------------------
    // 版本信息函数
    // ------------------------------------------------------------------------
    VersionInfo getVersionInfo();
    const char* getVersionString();
    int getVersionMajor();
    int getVersionMinor();
    int getVersionPatch();

    // ------------------------------------------------------------------------
    // 配置文件读取函数
    // ------------------------------------------------------------------------
    DetectionResultCode readROIFromConfig(const std::string& configPath, ROI& roi);
    DetectionResultCode readQuadROIFromConfig(const std::string& configPath, QuadROI& quadRoi);

// ------------------------------------------------------------------------
    // 工具函数
    // ------------------------------------------------------------------------
    std::string generateFileName(const std::string& basePath, const std::string& sn);

    void saveResultImage(const cv::Mat& sourceImage, const std::string& message, const std::string& sn,
                        const QuadROI& quadRoi, const std::string& outputDir, LidarDetectionResult &result);

    // ------------------------------------------------------------------------
    // 激光线检测核心函数
    // ------------------------------------------------------------------------
    LidarDetectionResult detectLidarLineWithQuadROI(const cv::Mat& image, const QuadROI& quadRoi, 
                                                   const std::string& sn, const std::string& outputDir);
    LidarLineResult detect(const cv::Mat& image, const std::string& configPath, 
                          const std::string& sn, const std::string& outputDir);

} // namespace LidarLineDetector

// ============================================================================
// C++ 命名空间：相机稳定性检测
// ============================================================================
namespace CameraStabilityDetection {
    
    // ------------------------------------------------------------------------
    // 相机自检相关函数
    // ------------------------------------------------------------------------
    DetectionResultCode loadTargetConfig(const std::string& configPath, 
                                        LidarLineDetector::TargetConfig& config);
    DetectionResultCode detectTargetCenter(const cv::Mat& image, cv::Point2f& outCenter, 
                                          cv::Mat& displayImage);
// 相机移动检测（新接口：内部处理图像保存）
    TargetMovementResult_C checkCameraMovement(const cv::Mat& image, 
                                              const std::string& configPath, 
                                              const std::string& outputDir = "");

} // namespace CameraStabilityDetection

// ============================================================================
// C++ 封装类：激光线检测器
// ============================================================================
class CLidarLineDetector {
private:
    LidarLineDetector::ROI m_roi;
    LidarLineDetector::QuadROI m_quadRoi;
    std::string m_sn, m_outputDir;

public:
    // ------------------------------------------------------------------------
    // 构造函数和析构函数
    // ------------------------------------------------------------------------
    CLidarLineDetector() = default;
    ~CLidarLineDetector() = default;

    // ------------------------------------------------------------------------
    // 初始化和配置函数
    // ------------------------------------------------------------------------
    DetectionResultCode initialize(const char* configPath);
    void setROI(int x, int y, int width, int height);
    void setSn(const char* sn);
    void setOutputDir(const char* outputDir);
// ------------------------------------------------------------------------
    // 激光线检测函数
    // ------------------------------------------------------------------------
    TLidarLineResult_C detect(const TCMat_C image, const char* configPath);
    
    // ------------------------------------------------------------------------
    // 相机自检相关方法
    // ------------------------------------------------------------------------
    DetectionResultCode loadTargetConfig(const char* configPath, 
                                        LidarLineDetector::TargetConfig& config);
    TargetMovementResult_C checkCameraStability(const TCMat_C image, const char* configPath);
    
    // ------------------------------------------------------------------------
    // 版本信息接口
    // ------------------------------------------------------------------------
    static Smpclass_API VersionInfo getVersionInfo();
    static Smpclass_API const char* getVersionString();
    static Smpclass_API int getVersionMajor();
    static Smpclass_API int getVersionMinor();
    static Smpclass_API int getVersionPatch();
};

// ============================================================================
// C接口函数声明
// ============================================================================
extern "C" {
    
    // ------------------------------------------------------------------------
    // 激光线检测器实例管理
    // ------------------------------------------------------------------------
    Smpclass_API CLidarLineDetector* CLidarLineDetector_new();
    Smpclass_API void CLidarLineDetector_delete(CLidarLineDetector* instance);

    // ------------------------------------------------------------------------
    // 激光线检测器配置函数
    // ------------------------------------------------------------------------
    Smpclass_API DetectionResultCode CLidarLineDetector_initialize(CLidarLineDetector* instance, 
                                                                  const char* configPath);
    Smpclass_API void CLidarLineDetector_setROI(CLidarLineDetector* instance, 
                                               int x, int y, int width, int height);
    Smpclass_API void CLidarLineDetector_setSn(CLidarLineDetector* instance, const char* sn);
    Smpclass_API void CLidarLineDetector_setOutputDir(CLidarLineDetector* instance, 
                                                     const char* outputDir);
    
    // ------------------------------------------------------------------------
    // 激光线检测函数
    // ------------------------------------------------------------------------
    Smpclass_API TLidarLineResult_C CLidarLineDetector_detect(CLidarLineDetector* instance, 
                                                             const TCMat_C image, 
                                                             const char* configPath);

// ------------------------------------------------------------------------
    // 相机自检相关C接口
    // ------------------------------------------------------------------------
    Smpclass_API DetectionResultCode CLidarLineDetector_loadTargetConfig(CLidarLineDetector* instance, 
                                                                        const char* configPath, 
                                                                        TTargetConfig_C* config);
    Smpclass_API TargetMovementResult_C CLidarLineDetector_checkCameraMovement(CLidarLineDetector* instance, 
                                                                             const TCMat_C image, 
                                                                             const char* configPath, 
                                                                             const char* outputDir);
    
    // ------------------------------------------------------------------------
    // 版本信息C接口
    // ------------------------------------------------------------------------
    Smpclass_API VersionInfo LidarLineDetector_GetVersionInfo();
    Smpclass_API const char* LidarLineDetector_GetVersionString();
    Smpclass_API int LidarLineDetector_GetVersionMajor();
    Smpclass_API int LidarLineDetector_GetVersionMinor();
    Smpclass_API int LidarLineDetector_GetVersionPatch();
}

#endif // LIDAR_LINE_DETECTION_H