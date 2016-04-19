#pragma once
#include "opencv2/opencv.hpp"

// Readable version of Di Stefano's algorithm
int DiStefano(const cv::Mat1b &img, cv::Mat1i &imgLabels);

// Optimized version of Di Stefano's algorithm
int DiStefanoOPT(const cv::Mat1b &img, cv::Mat1i &imgLabels);