#ifndef YACCLAB_LATEX_GENERATOR_H_
#define YACCLAB_LATEX_GENERATOR_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void GenerateLatexTable(const std::string& output_path, const std::string& latex_file, const cv::Mat1d& all_res,
    const std::vector<cv::String>& alg_name, const std::vector<cv::String>& ccl_algorithms);

void GenerateMemoryLatexTable(const std::string& output_path, const std::string& latex_file, const cv::Mat1d& accesses,
    const std::string& dataset, const std::vector<cv::String>& ccl_mem_algorithms);

void GenerateLatexCharts(const std::string& output_path, const std::string& latex_charts, const std::string& latex_folder,
    const std::vector<cv::String>& datasets_name);

#endif // !YACCLAB_LATEX_GENERATOR_H_