#pragma once
#include "opencv2/opencv.hpp"

// Optimized version of Chang's algorithm ( with contour tracing ) 
int CT_OPT(const cv::Mat1b &img, cv::Mat1i &imgLabels);
