#pragma once
#include "opencv2/opencv.hpp"

// Readable version of Zhao 2010 algorithm
int SBLA(const cv::Mat1b &img, cv::Mat1i &imgLabels);

int SBLA_OPT(const cv::Mat1b &img, cv::Mat1i &imgLabels);
