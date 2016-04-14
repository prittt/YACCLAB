#pragma once

#include "opencv2/opencv.hpp"

#include <vector>
#include <string>
#include <utility>      // std::pair, std::make_pair

// ADD HERE YOUR ".h" FILE          <===============================================================================
#include "labelingGranaOPT.h"
#include "labelingGranaOPTv2.h"
#include "labelingGranaOPTv3.h"
#include "labelingLightSPEED.h"
#include "labelingLightSPEEDold.h"
// Block-Based Connected-Component Labeling Algorithm Using Binary Decision Trees: Wan-Yu, Chung-Cheng Chiu and Jia-Horng Yang
#include "labelingChang2015.h"
#include "labelingOpenCv.h"

// FUNCTION POINTER: 
//	Mat1b:	the 8-bit single-channel image to be labeled;
//	Mat1i:	destination labeled image; 
typedef int(*CCLPointer) (const cv::Mat1b&, cv::Mat1i&);

// Map of connected components algorithms: function's name (first), pointer to algorithm (second)
#define new_algorithm(x) {#x, x}

//// ADD HERE YOUR FUNCTION NAME      <===============================================================================
//std::vector<std::pair<CCLPointer, std::string>> CCLAlgorithms = 
//{
//    new_algorithm(labelingGranaOPT),
//    new_algorithm(Wu),
//    new_algorithm(labelingGranaOPTv2),
//    new_algorithm(labelingGranaOPTv3),
//    new_algorithm(LSL_STD),
//    new_algorithm(CCIT)
//};

std::map<std::string, CCLPointer> CCLAlgorithmsMap =
{
    new_algorithm(labelingGranaOPT),
    new_algorithm(Wu),
    new_algorithm(labelingGranaOPTv2),
    new_algorithm(labelingGranaOPTv3),
    new_algorithm(LSL_STD),
    new_algorithm(CCIT)
};
// Map of connected components algorithms: functions's name (first), pointer to algorithm (second)