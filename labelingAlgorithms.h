#pragma once

#include "opencv2/opencv.hpp"

#include <string>
#include <utility>      // std::pair, std::make_pair

// ADD HERE YOUR ".h" FILE          <===============================================================================
// Optimized Block-based Connected Components Labeling with Decision Trees: Grana Costantino, Borghesani Daniele, Cucchiara Rita
#include "labelingGranaOPT.h"
#include "labelingGranaOPTv2.h"
#include "labelingGranaOPTv3.h"     
// Light speed labeling : efficient connected component labeling on RISC architectures: Lacassagne, Lionel and Zavidovique, Bertrand
#include "labelingLacassagne2011.h" 
#include "labelingLacassagne2011Readable.h"
// A component-labeling algorithm using contour tracing technique: F.Chang and C.Chen
#include "labelingChang2003.h"
// Block-Based Connected-Component Labeling Algorithm Using Binary Decision Trees: Wan-Yu, Chung-Cheng Chiu and Jia-Horng Yang
#include "labelingChang2015.h"      
// Optimizing two-pass connected-component labeling algorithms: Wu, Kesheng and Otoo, Ekow and Suzuki, Kenji
#include "labelingOpenCv.h"         
#include "LabelingDiStefano.h"

// FUNCTION POINTER: 
//	Mat1b:	the 8-bit single-channel image to be labeled;
//	Mat1i:	destination labeled image; 
typedef int(*CCLPointer) (const cv::Mat1b&, cv::Mat1i&);

// Map of connected components algorithms: function's name (first), pointer to algorithm (second)
#define new_algorithm(x) {#x, x}

// ADD HERE YOUR FUNCTION NAME      <===============================================================================
std::map<std::string, CCLPointer> CCLAlgorithmsMap =
{
    new_algorithm(labelingGranaOPT),
    new_algorithm(Wu),
    new_algorithm(labelingGranaOPTv2),
    new_algorithm(labelingGranaOPTv3),
    new_algorithm(LSL_STD),
    new_algorithm(CCIT),
	new_algorithm(labelingDiStefano),
	new_algorithm(labelingDiStefanoOPT),
	new_algorithm(CT)
};
// Map of connected components algorithms: functions's name (first), pointer to algorithm (second)