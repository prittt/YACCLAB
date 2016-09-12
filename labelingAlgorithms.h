// Copyright(c) 2016 - Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
// 
// * Neither the name of YACCLAB nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
#include "labelingFChang2003.h"
// Block-Based Connected-Component Labeling Algorithm Using Binary Decision Trees: Wan-Yu, Chung-Cheng Chiu and Jia-Horng Yang
#include "labelingWYChang2015.h"      
// Optimizing two-pass connected-component labeling algorithms: Wu, Kesheng and Otoo, Ekow and Suzuki, Kenji
#include "labelingWu2009.h"
#include "labelingWu2009OpenCV.h"
// A simple and efficient connected components labeling algorithm: L.Di Stefano and A.Bulgarelli
#include "labelingDiStefano1999.h"
// Configuration-Transition-Based Connected-Component Labeling: L. He, X. Zhao, Y.Chao, K. Suzuki
#include "labelingHe2014.h"
// Stripe-based connected components labelling: H.L. Zhao, Y.B. Fan, T.X. Zhang, H.S. Sang
#include "labelingZhao2010.h"
// A reference function which DOESN'T perform labeling, but allocates memory for output and scans the input image
// writing an output value (quasi random) to the labels image
#include "labelingNULL.h"
// Optimized Connected Components Labeling with Pixel Prediction
#include "labelingPred.h"

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
	new_algorithm(SAUFCV_OPT),
    // Grana
    new_algorithm(BBDT_OPT),
    new_algorithm(PRED_OPT),
    // Lacassagne
    new_algorithm(LSL_STD),
    new_algorithm(LSL_STD_OPT),
	// He
	new_algorithm(CTB_OPT),
    // Fu Chang
    new_algorithm(CT_OPT),
	// Wan-Yu Chang
    new_algorithm(CCIT_OPT),
	// Zhao
	new_algorithm(SBLA),
	new_algorithm(SBLA_OPT),
	// NULL
	new_algorithm(labelingNULL)
};
// Map of connected components algorithms: functions's name (first), pointer to algorithm (second)