#ifndef YACCLAB_YACCLAB_TESTS_H_
#define YACCLAB_YACCLAB_TESTS_H_
#include "file_manager.h"

#include <map>

#include <opencv2/imgproc.hpp>

#include "config_data.h"
#include "file_manager.h"
#include "labeling_algorithms.h"

class YacclabTests {
public:
    YacclabTests(const ConfigData& cfg) : cfg_(cfg) {}

    void CheckPerformLabeling();
    void CheckPerformLabelingWithSteps();
    void CheckPerformLabelingMem();

    void CheckAlgorithms(); // TODO move private

    void AverageTest();
    void AverageTestWithSteps();

    //TODO: Check correctness of memory tests also

    //Other test related functions
    void LatexGenerator();
private:
    ConfigData cfg_;
    cv::Mat1d average_results_;
    std::map<cv::String, cv::Mat1d> average_ws_results_;
    std::map<std::string, cv::Mat1d> memory_accesses_;

    bool LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const filesystem::path& files_path);
    void SaveBroadOutputResults(std::map<cv::String, cv::Mat1d>& results, const std::string& o_filename, const cv::Mat1i& labels, const std::vector<std::pair<std::string, bool>>& filenames);
    void SaveBroadOutputResults(const cv::Mat1d& results, const std::string& o_filename, const cv::Mat1i& labels, const std::vector<std::pair<std::string, bool>>& filenames);
};

#endif // !YACCLAB_YACCLAB_TESTS_H_
