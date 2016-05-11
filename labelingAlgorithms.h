#pragma once

#include "opencv2/opencv.hpp"

#include <string>
#include <utility>      // std::pair, std::make_pair

// ADD HERE YOUR ".h" FILE          <===============================================================================
// Optimized Block-based Connected Components Labeling with Decision Trees: Grana Costantino, Borghesani Daniele, Cucchiara Rita
#include "labelingGrana2010.h"    
// Light speed labeling : efficient connected component labeling on RISC architectures: Lacassagne, Lionel and Zavidovique, Bertrand
#include "labelingLacassagne2011.h" 
// A component-labeling algorithm using contour tracing technique: F.Chang and C.Chen
#include "labelingChang2003.h"
// Block-Based Connected-Component Labeling Algorithm Using Binary Decision Trees: Wan-Yu, Chung-Cheng Chiu and Jia-Horng Yang
#include "labelingChang2015.h"      
// Optimizing two-pass connected-component labeling algorithms: Wu, Kesheng and Otoo, Ekow and Suzuki, Kenji
#include "labelingWu2009.h"
// A simple and efficient connected components labeling algorithm: L.Di Stefano and A.Bulgarelli
#include "labelingDiStefano1999.h"

#include "labelingZhao2010.h"

#include "labelingNULL.h"

// FUNCTION POINTER: 
//	Mat1b:	the 8-bit single-channel image to be labeled;
//	Mat1i:	destination labeled image; 
typedef int(*CCLPointer) (const cv::Mat1b&, cv::Mat1i&);

// Map of connected components algorithms: function's name (first), pointer to algorithm (second)
#define new_algorithm(x) {#x, x}

// ADD HERE YOUR FUNCTION NAME      <===============================================================================
std::map<std::string, CCLPointer> CCLAlgorithmsMap =
{
    // Di Stefano
    new_algorithm(DiStefano),
    new_algorithm(DiStefanoOPT),
    // Wu/He
    new_algorithm(SAUF_OPT),
    // Grana
    //new_algorithm(BBDT),
    new_algorithm(BBDT_OPT),
    // Lacassagne
    new_algorithm(LSL_STD),
    new_algorithm(LSL_STD_OPT),
    // Chang
    new_algorithm(CT_OPT),
    new_algorithm(CCIT_OPT),
	// Zhao
	new_algorithm(SBLA),
	new_algorithm(SBLA_OPT),
	// NULL
	new_algorithm(labelingNULL)
};
// Map of connected components algorithms: functions's name (first), pointer to algorithm (second)