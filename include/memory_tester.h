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

#ifndef MEMORY_TESTER_H_
#define MEMORY_TESTER_H_

#include "opencv2/core.hpp"

enum memorydatatype {
    // Data structures for "classical" algorithms
    MD_BINARY_MAT = 0,
    MD_LABELED_MAT = 1,
    MD_EQUIVALENCE_VEC = 2,

    // Other data_ structures
    MD_OTHER = 3,

    // Total number of data_ structures in the list
    MD_SIZE = 4,
};

template <typename T>
class MemMat {
public:

    int rows;
    int cols;

	MemMat() {}

    MemMat(cv::Mat_<T> img)
    {
        img_ = img.clone(); // Deep copy
        accesses_ = cv::Mat1i(img.size(), 0);
        rows = img.rows;
        cols = img.cols;
    }

    MemMat(cv::Size size)
    {
        img_ = cv::Mat_<T>(size);
        accesses_ = cv::Mat1i(size, 0);
        rows = size.height;
        cols = size.width;
    }

    MemMat(int rows, int cols) : MemMat(cv::Size(cols, rows)) {}

    MemMat(cv::Size size, const T val)
    {
        img_ = cv::Mat_<T>(size, val);
        accesses_ = cv::Mat1i(size, 1);	// The initialization accesses must be counted
        rows = size.height;
        cols = size.width;
    }

    MemMat(int rows, int cols, const T val) : MemMat(cv::Size(cols, rows), val) {}

    T& operator()(const int r, const int c)
    {
        accesses_.ptr<int>(r)[c]++; // Count access
        return img_.template ptr<T>(r)[c];
    }

    T& operator()(const int x)
    {
        accesses_(x)++; // Count access
        return img_(x);
    }

    cv::Mat_<T> GetImage() const
    {
        return img_.clone();
    }

    cv::Mat1i GetAccessesMat() const
    {
        return accesses_.clone();
    }

    double GetTotalAccesses() const
    {
        return cv::sum(accesses_)[0];
    }

    //~MemMat();  // This is the destructor: declaration

private:
    cv::Mat_<T> img_;
    cv::Mat1i accesses_;
};

template <typename T>
class MemVector {
public:
	MemVector() {}

    MemVector(std::vector<T> vec)
    {
        vec_ = vec;  // Deep copy
        accesses_ = std::vector<int>(vec.size(), 0);
    }

    MemVector(const size_t size)
    {
        vec_ = std::vector<T>(size);
        accesses_ = std::vector<int>(size, 0);
    }

    MemVector(const size_t size, const T val)
    {
        vec_ = std::vector<T>(size, val);
        accesses_ = std::vector<int>(size, 1); // The initilization accesses must be counted
    }

    T& operator[](const int i)
    {
        accesses_[i]++; // Count access
        return vec_[i];
    }

    std::vector<T> GetVector() const
    {
        return vec_;
    }

    std::vector<T> GetAccessesVector() const
    {
        return accesses_;
    }

    double GetTotalAccesses() const
    {
        double tot = 0;
        for (size_t i = 0; i < accesses_.size(); ++i)
        {
            tot += accesses_[i];
        }

        return tot;
    }

    T* GetDataPointer()
    {
        return vec_.data_();
    }

    size_t size()
    {
        return vec_.size();
    }

    void Memiota(size_t begin, size_t end, const T value)
    {
        T _value = value;

        for (size_t i = begin; i < end; ++i)
        {
            vec_[i] = _value++;
            accesses_[i]++;	// increment access
        }
    }

    //~MemVector();  // This is the destructor: declaration

private:
    std::vector<T> vec_;
    std::vector<int> accesses_;
};

template <typename T>
class MemVol {
public:

    int w;
    int h;
    int d;

    MemVol() {}

    MemVol(cv::Mat_<T> img)
    {
        if (img.dims != 3) {
            throw std::runtime_error("Illegal dims field, it should be 3.");
        }
        img_ = img.clone(); // Deep copy

        accesses_ = cv::Mat1i(3, img.size.p, 0);
        w = img.size[2];
        h = img.size[1];
        d = img.size[0];
    }

    MemVol(const int* sizes)
    {
        img_ = cv::Mat_<T>(3, sizes);
        accesses_ = cv::Mat1i(3, img_.size.p, 0);
        w = sizes[2];
        h = sizes[1];
        d = sizes[0];
    }

    MemVol(int w, int h, int d)
    {
        int sizes[3] = { d, h, w };
        img_ = cv::Mat_<T>(sizes);
        accesses_ = cv::Mat1i(3, sizes, 0);
        w = sizes[2];
        h = sizes[1];
        d = sizes[0];
    }

    MemVol(const int* sizes, const T val)
    {
        img_ = cv::Mat_<T>(3, sizes, val);
        accesses_ = cv::Mat_<int>(3, sizes, 1);	// The initialization accesses must be counted
        w = sizes[2];
        h = sizes[1];
        d = sizes[0];
    }

    MemVol(int w, int h, int d, const T val)
    {
        int sizes[3] = { d, h, w };
        img_ = cv::Mat_<T>(3, sizes, val);
        accesses_ = cv::Mat1i(3, sizes, 1);
        w = sizes[2];
        h = sizes[1];
        d = sizes[0];
    }

    T& operator()(int s, int r, int c)
    {
        (*accesses_.ptr<int>(s, r, c))++;   // Count access

        return *img_.template ptr<T>(s, r, c);
    }

    T& operator()(const int x)
    {
        accesses_(x)++; // Count access
        return img_(x);
    }

    cv::Mat_<T> GetImage() const
    {
        return img_.clone();
    }

    cv::Mat1i GetAccessesMat() const
    {
        return accesses_.clone();
    }

    double GetTotalAccesses() const
    {
        return cv::sum(accesses_)[0];
    } 

    //~MemVol();  // This is the destructor: declaration

public: // DEBUG
    cv::Mat_<T> img_;
    cv::Mat1i accesses_;
};

#endif // MEMORY_TESTER_H_