#pragma once
#include "opencv2/opencv.hpp"

int labelingDiStefano(const cv::Mat1b &img, cv::Mat1i &imgLabels);
int labelingDiStefanoOPT(const cv::Mat1b &img, cv::Mat1i &imgLabels);