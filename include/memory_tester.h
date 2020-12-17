// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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