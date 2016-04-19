// Block-Based Connected-Component Labeling Algorithm Using Binary Decision Trees
// Copyright (C) 2015 - Wan-Yu Chang and Chung-Cheng Chiu
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free 
// Software Foundation; either version 3 of the License, or (at your option) 
// any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
// details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this library; if not, see <http://www.gnu.org/licenses/>.
// 
// The "free" of library only licensed for the purposes of research and study. 
// For further information or business co-operation, please contact us at
// Wan-Yu Chang and Chung-Cheng Chiu - Chung Cheng Institute of Technology of National Defense University.
// No.75, Shiyuan Rd., Daxi Township, Taoyuan County 33551, Taiwan (R.O.C.)  - e-mail: david.cc.chiu@gmail.com 
//
// Specially thank for the help of Prof. Grana who provide his source code of the BBDT algorithm.

#pragma once
#include "opencv2/opencv.hpp"

// Optimized version of Chang's algorithm ( block based ) 
int CCIT_OPT(const cv::Mat1b &img, cv::Mat1i &imgLabels);
