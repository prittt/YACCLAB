#pragma once
#include "opencv2/opencv.hpp"

#include "performanceEvaluator.h"

// Readable version of Zhao 2010 algorithm
int SBLAmio(const cv::Mat1b &img, cv::Mat1i &imgLabels);

int SBLAmioOPT(const cv::Mat1b &img, cv::Mat1i &imgLabels);

// Authors version of Zhao 2010 algorithm
int SBLA(const cv::Mat1b &img, cv::Mat1i &imgLabels);

int SBLA_perf(const cv::Mat1b &img, cv::Mat1i &imgLabels, PerformanceEvaluator& perf);

// Optimized version of Di Stefano's algorithm
//int DiStefanoOPT(const cv::Mat1b &img, cv::Mat1i &imgLabels);