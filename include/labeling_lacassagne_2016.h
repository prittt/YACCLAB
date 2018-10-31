// Copyright(c) 2016 - 2018 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

#ifndef YACCLAB_LABELING_LACASSAGNE_2011_H_
#define YACCLAB_LABELING_LACASSAGNE_2011_H_

#include <numeric>

#include <opencv2/core.hpp>

#include "labels_solver.h"
#include "labeling_algorithms.h"
#include "memory_tester.h"

using namespace std;
using namespace cv;

struct Table2D {
    unsigned **prows = nullptr;

    Table2D() {}
    Table2D(const Table2D&) = delete;
    Table2D& operator=(const Table2D&) = delete;
    Table2D(size_t rows, size_t cols) { Reserve(rows, cols); }
    ~Table2D() { Release(); }

    void Reserve(size_t rows, size_t cols) {
        prows = new unsigned*[rows];
        *prows = new unsigned[rows*cols];
        unsigned *currow = *prows;
        for (size_t i = 1; i < rows; ++i) {
            currow += cols;
            prows[i] = currow;
        }
    }

    void Release() {
        if (prows) {
            delete[] *prows;
            delete[] prows;
            prows = nullptr;
        }
    }
};

// Modifications for RLE version
#define RLE_MOD1 //if (f) 
#define RLE_MOD2 f

// Modifications for zero-offset addressing
#define ZOA_MOD0 int b = 0; // Right border compensation
#define ZOA_MOD1 - b
#define ZOA_MOD2 b = b ^ RLE_MOD2;
#define ZOA_MOD3 x0 = 0; f = x0 ^ x1;
#define ZOA_MOD4 er = er + f;
#define ZOA_MOD5 

template <typename LabelsSolver>
class LSL_STD : public Labeling2D<CONN_8> {
#include "labeling_lacassagne_2016_code.inc"
};

#undef ZOA_MOD0
#undef ZOA_MOD1
#undef ZOA_MOD2
#undef ZOA_MOD3
#undef ZOA_MOD4
#undef ZOA_MOD5
#undef RLE_MOD1
#undef RLE_MOD2


#define RLE_MOD1 //if (f) 
#define RLE_MOD2 f
#define ZOA_MOD0 
#define ZOA_MOD1 
#define ZOA_MOD2 
#define ZOA_MOD3 if (x1 != 0) {
#define ZOA_MOD4 } er = er + x1;
#define ZOA_MOD5 -1

template <typename LabelsSolver>
class LSL_STDZ : public Labeling2D<CONN_8> {
#include "labeling_lacassagne_2016_code.inc"
};

#undef ZOA_MOD0
#undef ZOA_MOD1
#undef ZOA_MOD2
#undef ZOA_MOD3
#undef ZOA_MOD4
#undef ZOA_MOD5
#undef RLE_MOD1
#undef RLE_MOD2

#define RLE_MOD1 if (f) 
#define RLE_MOD2 1
#define ZOA_MOD0 int b = 0; // Right border compensation
#define ZOA_MOD1 - b
#define ZOA_MOD2 b = b ^ RLE_MOD2;
#define ZOA_MOD3 x0 = 0; f = x0 ^ x1;
#define ZOA_MOD4 er = er + f;
#define ZOA_MOD5 

template <typename LabelsSolver>
class LSL_RLE : public Labeling2D<CONN_8> {
#include "labeling_lacassagne_2016_code.inc"
};

#undef ZOA_MOD0
#undef ZOA_MOD1
#undef ZOA_MOD2
#undef ZOA_MOD3
#undef ZOA_MOD4
#undef ZOA_MOD5
#undef RLE_MOD1
#undef RLE_MOD2

#define RLE_MOD1 if (f) 
#define RLE_MOD2 1
#define ZOA_MOD0 
#define ZOA_MOD1 
#define ZOA_MOD2 
#define ZOA_MOD3 if (x1 != 0) {
#define ZOA_MOD4 } er = er + x1;
#define ZOA_MOD5 -1

template <typename LabelsSolver>
class LSL_RLEZ : public Labeling2D<CONN_8> {
#include "labeling_lacassagne_2016_code.inc"
};

#undef ZOA_MOD0
#undef ZOA_MOD1
#undef ZOA_MOD2
#undef ZOA_MOD3
#undef ZOA_MOD4
#undef ZOA_MOD5
#undef RLE_MOD1
#undef RLE_MOD2

#endif // !YACCLAB_LABELING_LACASSAGNE_2011_H_
