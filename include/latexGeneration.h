#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

void generateLatexTable(const std::string& output_path, const std::string& latex_file, const cv::Mat1d& all_res,
    const std::vector<cv::String>& algName, const std::vector<cv::String>& CCLAlgorithms);

void generateMemoryLatexTable(const std::string& output_path, const std::string& latex_file, const cv::Mat1d& accesses,
    const std::string& dataset, const std::vector<cv::String>& CCLMemAlgorithms);

void generateLatexCharts(const std::string& output_path, const std::string& latex_charts, const std::string& latex_folder,
    const std::vector<cv::String>& datasetsName);
