#pragma once
#include "opencv2/opencv.hpp"

// Readable version of Lacassagne's algorithm
int LSL_STD(const cv::Mat1b& img,cv::Mat1i& labels);

// Optimized version of Lacassagne's algorithm
int LSL_STD_OPT(const cv::Mat1b& img, cv::Mat1i& labels);