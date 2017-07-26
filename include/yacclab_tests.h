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
    YacclabTests(const ConfigData& cfg) : cfg_(cfg)
    {
        average_results_ = cv::Mat1d(cfg_.average_datasets.size(), cfg_.ccl_algorithms.size(), std::numeric_limits<double>::max());
        average_ws_results_ = cv::Mat1d(cfg_.average_datasets.size(), cfg_.ccl_algorithms.size() * StepType::ST_SIZE, std::numeric_limits<double>::max());
    }

    void CheckPerformLabeling();
    void CheckPerformLabelingWithSteps();
    void CheckPerformLabelingMem();

    void CheckAlgorithms(); // TODO move private

    void AverageTest(cv::Mat1d& average_results_);
    void AverageTestWithSteps(cv::Mat1d& average_ws_results_);

    //TODO: Check correctness of memory tests also

    //Other test functions

private:
    ConfigData cfg_;
    cv::Mat1d average_results_;
    cv::Mat1d average_ws_results_;
    std::map<std::string, cv::Mat1d> memory_accesses_;


    bool LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const filesystem::path& files_path);

    void SaveBroadOutputResults(std::map<cv::String, cv::Mat1d>& results, const std::string& o_filename, const cv::Mat1i& labels, const std::vector<std::pair<std::string, bool>>& filenames);
    void SaveBroadOutputResults(const cv::Mat1d& results, const std::string& o_filename, const cv::Mat1i& labels, const std::vector<std::pair<std::string, bool>>& filenames);

    void LatexGenerator();
};

#endif // !YACCLAB_YACCLAB_TESTS_H_
