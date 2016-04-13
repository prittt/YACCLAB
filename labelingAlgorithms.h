#pragma once

#include <vector>
#include <string>
#include <utility>      // std::pair, std::make_pair

#include "opencv2/opencv.hpp"

// ADD HERE YOUR ".h" FILE          <===============================================================================
#include "labelingGranaOPT.h"
#include "labelingLightSPEED.h"
#include "labelingOpenCv.h"
#include "labelingGranaOPTv2.h"
#include "labelingGranaOPTv3.h"
#include "labelingLightSPEEDold.h"
// Block-Based Connected-Component Labeling Algorithm Using Binary Decision Trees: Wan-Yu, Chung-Cheng Chiu and Jia-Horng Yang
#include "labelingChang2015.h"


// FUNCTION POINTER: 
//	Mat1b:	the 8-bit single-channel image to be labeled;
//	Mat1i:	destination labeled image; 
typedef int(*CCLPointer) (const cv::Mat1b&, cv::Mat1i&);

// Vector of connected component algorithms: pointer to algorithm (first), algorithms's name (second)
#define new_algorithm(x) {x, #x}

// ADD HERE YOUR FUNCTION NAME      <===============================================================================
std::vector<std::pair<CCLPointer, std::string>> CCLAlgorithms = 
{
    new_algorithm(labelingGranaOPT),
    new_algorithm(Wu),
    new_algorithm(labelingGranaOPTv2),
    new_algorithm(labelingGranaOPTv3),
    new_algorithm(LSL_STD),
    new_algorithm(CCIT)
    //new_algorithm(labelingOPENCV)
};
// Vector of connected component algorithms: pointer to algorithm (first), algorithms's name (second)
