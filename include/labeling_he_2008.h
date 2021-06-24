// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_HE_2008_H_
#define YACCLAB_LABELING_HE_2008_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

// This Circular Buffer implementation is required for the Run Based algorithms introduced by He.
template<typename T>
class CircularBuffer
{
public:
    T* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t size_ = 0;
    size_t tail_ = 0;
    size_t head_ = 0;

    CircularBuffer() = delete;

    CircularBuffer(size_t capacity) : buffer_{ new T[capacity] }, capacity_{ capacity } {}

    ~CircularBuffer()
    {
        delete[] buffer_;
    }

    // Adds new element to the circular buffer
    T Enqueue(T item)
    {
        //if (IsFull()) {
        //    std::cout << "Why are you calling Enqueue on a full Circular Buffer?";
        //    throw std::runtime_error("No space left in the Circular Buffer!");
        //}

        size_++;
        buffer_[tail_] = item;
        tail_ = (tail_ + 1) % capacity_;
        return item;
    }

    // Gets the head of the circular buffer without removing it
    T Front()
    {
        //if (IsEmpty()) {
        //    std::cout << "Why are you calling CircularBuffer::Front() on an empty queue?";
        //    throw std::runtime_error("The Circular Buffer is empty!");
        //}

        return buffer_[head_];
    }

    // Gets the head of the circular buffer and removes it from the buffer
    T Dequeue()
    {
        size_--;
        T element = Front();
        head_ = (head_ + 1) % capacity_;
        return element;
    }

    // Removes the head of the circular buffer and returns the next head 
    T DequeueAndFront()
    {
        Dequeue();
        return Front();
    }

    bool IsFull() { return size_ == capacity_; }
    bool IsEmpty() { return !size_; }

};

template <typename LabelsSolver>
class RBTS : public Labeling2D<Connectivity2D::CONN_8>
{
public:
    RBTS() {}

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size(), 0); // Call to memset

        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();

        // Scan Mask (Rosenfeld)
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+
        
        // First Scan
        int w(img_.cols); // w = N in the paper naming convention
        int h(img_.rows);

        // Circular Buffers
        int buffer_capacity = w / 2 + 2;
        CircularBuffer<int> s_queue(buffer_capacity);
        CircularBuffer<int> e_queue(buffer_capacity);

        for (int r = 0; r < h; r += 1) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            // const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; c += 1) {

                // Is there a new run ?
                if (img_row[c] == 0) {
                    continue;
                }

                // clb (consider left border) and crb (consider right border) are used to handle border cases ignored by
                // the original paper.

                // Yes (new run)
                // 1) We need to record the new run r(s,e)
                int clb = c != 0; // Consider left border
                int s = s_queue.Enqueue(w * r + c++); // Current run starting index
                for (; c < w && img_row[c] > 0; c += 1) {}
                int crb = c != w; // Consider right border (derived by c - 1 != w - 1)
                int e = e_queue.Enqueue(w * r + --c); // Current run end index

                // 2) Discard all the runs until a run r(h,t) that end after/at s - N - 1 (t in the following scheme)
                // +-+-+-+-+-+-+-+-+-+
                // | |t|>| | | | | | |
                // +-+-+-+-+-+-+-+-+-+
                // | | |s| | | |e| | |
                // +-+-+-+-+-+-+-+-+-+
                // This run always exists. It is at most the run we just found.
                int h = s_queue.Front();
                int t = e_queue.Front();
                while (t < s - w - clb) {
                    h = s_queue.DequeueAndFront();
                    t = e_queue.DequeueAndFront();
                }
                
                // 3A) If the run r(h,t) starts before/at e - N + 1 (h in the following scheme)
                // +-+-+-+-+-+-+-+-+-+
                // | | | | | | |<|h| |
                // +-+-+-+-+-+-+-+-+-+
                // | | |s| | | |e| | |
                // +-+-+-+-+-+-+-+-+-+
                // it connects to the current run so we update the label of the current run
                // with that of the run(h,t), taking the value from p(h) -> p(r - 1, h % w)
                // The condition (e - w + 1) % w != 0 that can be simplified as (e + 1) % w != 0 
                // is for the handling of (left) border cases that were ignored by the original 
                // paper in which the border is always considered black (background).
                bool next = false;
                if (h <= e - w + crb) {
                    unsigned val = img_labels_row_prev[h % w];
                    unsigned start = s % w;
                    for (unsigned i = start; i < start + e - s + 1; ++i) {
                        img_labels_row[i] = val;
                    }

                    // 3B) Moreover, we check for all the following run that end before/at e - N (t
                    // in the following schema), we perform a label merge of that run with the
                    // current one and we remove them from the queues (no other following run can
                    // be directly connected with these).
                    // +-+-+-+-+-+-+-+-+-+
                    // | | | | | |<|t| | |
                    // +-+-+-+-+-+-+-+-+-+
                    // | | |s| | | |e| | |
                    // +-+-+-+-+-+-+-+-+-+
                    // The label of the current run is val. The label of the run we are analyzing 
                    // from the queue is found at p(h) -> p(r - 1, h % w).
                    while (t <= e - w) {
                        if (next) {
                            LabelsSolver::Merge(img_labels_row_prev[h % w], val);
                        }
                        next = true;
                        h = s_queue.DequeueAndFront();
                        t = e_queue.DequeueAndFront();
                    }

                    // 4) We check for the next run which ends after/at e - N + 1 (t in the following
                    // schema. It always exists and it is at most the current run.
                    // +-+-+-+-+-+-+-+-+-+
                    // | | | | | | | |t|>|
                    // +-+-+-+-+-+-+-+-+-+
                    // | | |s| | | |e| | |
                    // +-+-+-+-+-+-+-+-+-+
                    while (t < e - w + 1) {
                        next = true;
                        s_queue.Dequeue();
                        e_queue.Dequeue();
                        h = s_queue.Front();
                        t = e_queue.Front();
                    }

                    // We must keep the run in the list but we may need to merge labels:
                    // If the run p(h, t) starts before/at e - N + 1 (h in the following
                    // schema) it connects to the current run so we need to merge labels.
                    // +-+-+-+-+-+-+-+-+-+
                    // | | | | | | |<|h| |
                    // +-+-+-+-+-+-+-+-+-+
                    // | | |s| | | |e| | |
                    // +-+-+-+-+-+-+-+-+-+
                    // The condition (e - w + 1) % w != 0 that can be simplified as (e + 1) % w != 0 
                    // is for the handling of (left) border cases that were ignored by the original 
                    // paper in which the border is always considered black (background).
                    if (h <= e - w + crb) {
                        if (next) { 
                            LabelsSolver::Merge(img_labels_row[s % w], img_labels_row_prev[h % w]);
                        }
                    }
                }
                else {
                    // Otherwise it is a run not connected to any other, so we need a new label
                    unsigned val = LabelsSolver::NewLabel();
                    unsigned start = s % w;
                    for (unsigned i = start; i < start + e - s + 1; ++i) {
                        img_labels_row[i] = val;
                    }
                }

            }//End columns's for
        }//End rows's for

        n_labels_ = LabelsSolver::Flatten();

        // Second scan
        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
            unsigned int *img_labels_row_start = img_labels_.ptr<unsigned int>(r_i);
            unsigned int *img_labels_row_end = img_labels_row_start + img_labels_.cols;
            unsigned int *img_labels_row = img_labels_row_start;
            for (int c_i = 0; img_labels_row != img_labels_row_end; ++img_labels_row, ++c_i) {
                *img_labels_row = LabelsSolver::GetLabel(*img_labels_row);
            }
        }

        LabelsSolver::Dealloc();
    }
    //    void PerformLabelingWithSteps()
    //    {
    //        double alloc_timing = Alloc();
    //
    //        perf_.start();
    //        FirstScan();
    //        perf_.stop();
    //        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());
    //
    //        perf_.start();
    //        SecondScan();
    //        perf_.stop();
    //        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());
    //
    //        perf_.start();
    //        Dealloc();
    //        perf_.stop();
    //        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);
    //
    //    }
    //    void PerformLabelingMem(std::vector<uint64_t>& accesses)
    //    {
    //        MemMat<unsigned char> img(img_);
    //        MemMat<int> img_labels(img_.size(), 0);
    //
    //        LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
    //        LabelsSolver::MemSetup();
    //
    //        // Scan Mask
    //        // +--+--+--+
    //        // |n1|n2|n3|
    //        // +--+--+--+
    //        // |n4|a |
    //        // +--+--+
    //        // |n5|b |
    //        // +--+--+
    //
    //        // First Scan
    //        int w(img_.cols);
    //        int h(img_.rows);
    //
    //        for (int r = 0; r < h; r += 2) {
    //
    //            int prob_fol_state = BR;
    //            int prev_state = BR;
    //
    //            for (int c = 0; c < w; c += 1) {
    //
    //                // A bunch of defines used to check if the pixels are foreground, and current state of graph
    //                // without going outside the image limits.
    //
    //#define CONDITION_A img(r,c)>0
    //#define CONDITION_B r+1<h && img(r + 1, c)>0
    //#define CONDITION_N1 c-1>=0 && r-1>=0 && img(r - 1, c - 1)>0
    //#define CONDITION_N2 r-1>=0 && img(r - 1, c)>0
    //#define CONDITION_N3 r-1>=0 && c+1<w && img(r - 1, c + 1)>0
    //#define CONDITION_N4 c-1>=0 && img(r, c - 1)>0
    //#define CONDITION_N5 c-1>=0 && r+1<h && img(r + 1, c - 1)>0
    //
    //#define ACTION_1 // nothing
    //#define ACTION_2 img_labels(r, c) = LabelsSolver::MemNewLabel(); // new label
    //#define ACTION_3 img_labels(r, c) = img_labels(r - 1, c - 1); // a <- n1
    //#define ACTION_4 img_labels(r, c) = img_labels(r - 1, c); // a <- n2
    //#define ACTION_5 img_labels(r, c) = img_labels(r - 1, c + 1); // a <- n3
    //#define ACTION_6 img_labels(r, c) = img_labels(r, c - 1); // a <- n4
    //#define ACTION_7 img_labels(r, c) = img_labels(r + 1, c - 1); // a <- n5
    //#define ACTION_8 LabelsSolver::MemMerge(img_labels(r,c), img_labels(r - 1, c - 1)); // a + n1
    //#define ACTION_9 LabelsSolver::MemMerge(img_labels(r,c), img_labels(r - 1, c + 1)); // a + n3
    //#define ACTION_10 LabelsSolver::MemMerge(img_labels(r,c), img_labels(r + 1, c - 1)); // a + n5
    //#define ACTION_11 img_labels(r + 1, c) = LabelsSolver::MemNewLabel(); // b new label
    //#define ACTION_12 img_labels(r + 1, c) = img_labels(r, c - 1); // b <- n4
    //#define ACTION_13 img_labels(r + 1, c) = img_labels(r + 1, c - 1); // b <- n5
    //#define ACTION_14 img_labels(r + 1, c) = img_labels(r, c); // b <- a
    //
    //#include "labeling_he_2014_graph.inc"
    //            }//End columns's for
    //        }//End rows's for
    //
    //#undef ACTION_1 
    //#undef ACTION_2 
    //#undef ACTION_3 
    //#undef ACTION_4 
    //#undef ACTION_5 
    //#undef ACTION_6 
    //#undef ACTION_7 
    //#undef ACTION_8 
    //#undef ACTION_9 
    //#undef ACTION_10
    //#undef ACTION_11
    //#undef ACTION_12
    //#undef ACTION_13
    //#undef ACTION_14
    //
    //#undef CONDITION_A 
    //#undef CONDITION_B 
    //#undef CONDITION_N1
    //#undef CONDITION_N2
    //#undef CONDITION_N3
    //#undef CONDITION_N4
    //#undef CONDITION_N5
    //
    //        n_labels_ = LabelsSolver::MemFlatten();
    //
    //        // Second scan
    //        for (int r_i = 0; r_i < img_labels.rows; ++r_i) {
    //            for (int c_i = 0; c_i < img_labels.cols; ++c_i) {
    //                img_labels(r_i, c_i) = LabelsSolver::MemGetLabel(img_labels(r_i, c_i));
    //            }
    //        }
    //
    //        // Store total accesses in the output vector 'accesses'
    //        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);
    //
    //        accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
    //        accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
    //        accesses[MD_EQUIVALENCE_VEC] = (uint64_t)LabelsSolver::MemTotalAccesses();
    //
    //        img_labels_ = img_labels.GetImage();
    //
    //        LabelsSolver::MemDealloc();
    //    }

private:
//    double Alloc()
//    {
//        // Memory allocation of the labels solver
//        double ls_t = LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY, perf_);
//        // Memory allocation for the output image
//        perf_.start();
//        img_labels_ = cv::Mat1i(img_.size());
//        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
//        perf_.stop();
//        double t = perf_.last();
//        perf_.start();
//        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
//        perf_.stop();
//        double ma_t = t - perf_.last();
//        // Return total time
//        return ls_t + ma_t;
//    }
//    void Dealloc()
//    {
//        LabelsSolver::Dealloc();
//        // No free for img_labels_ because it is required at the end of the algorithm 
//    }
//    void FirstScan()
//    {
//        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart); // Initialization
//        LabelsSolver::Setup();
//
//        // Scan Mask
//        // +--+--+--+
//        // |n1|n2|n3|
//        // +--+--+--+
//        // |n4|a |
//        // +--+--+
//        // |n5|b |
//        // +--+--+
//
//        // First Scan
//        int w(img_.cols);
//        int h(img_.rows);
//
//        for (int r = 0; r < h; r += 2) {
//            // Get rows pointer
//            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
//            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
//            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
//            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
//            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);
//            unsigned int* const img_labels_row_fol = (unsigned int *)(((char *)img_labels_row) + img_labels_.step.p[0]);
//
//            int prob_fol_state = BR;
//            int prev_state = BR;
//
//            for (int c = 0; c < w; c += 1) {
//
//                // A bunch of defines used to check if the pixels are foreground, and current state of graph
//                // without going outside the image limits.
//
//#define CONDITION_A img_row[c]>0
//#define CONDITION_B r+1<h && img_row_fol[c]>0
//#define CONDITION_N1 c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
//#define CONDITION_N2 r-1>=0 && img_row_prev[c]>0
//#define CONDITION_N3 r-1>=0 && c+1<w && img_row_prev[c+1]>0
//#define CONDITION_N4 c-1>=0 && img_row[c-1]>0
//#define CONDITION_N5 c-1>=0 && r+1<h && img_row_fol[c-1]>0
//
//#define ACTION_1 // nothing
//#define ACTION_2 img_labels_row[c] = LabelsSolver::NewLabel(); // new label
//#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // a <- n1
//#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // a <- n2
//#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // a <- n3
//#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // a <- n4
//#define ACTION_7 img_labels_row[c] = img_labels_row_fol[c - 1]; // a <- n5
//#define ACTION_8 LabelsSolver::Merge(img_labels_row[c], img_labels_row_prev[c - 1]); // a + n1
//#define ACTION_9 LabelsSolver::Merge(img_labels_row[c], img_labels_row_prev[c + 1]); // a + n3
//#define ACTION_10 LabelsSolver::Merge(img_labels_row[c], img_labels_row_fol[c - 1]); // a + n5
//#define ACTION_11 img_labels_row_fol[c] = LabelsSolver::NewLabel(); // b new label
//#define ACTION_12 img_labels_row_fol[c] = img_labels_row[c - 1]; // b <- n4
//#define ACTION_13 img_labels_row_fol[c] = img_labels_row_fol[c - 1]; // b <- n5
//#define ACTION_14 img_labels_row_fol[c] = img_labels_row[c]; // b <- a
//
//#include "labeling_he_2014_graph.inc"
//            }//End columns's for
//        }//End rows's for
//
//#undef ACTION_1 
//#undef ACTION_2 
//#undef ACTION_3 
//#undef ACTION_4 
//#undef ACTION_5 
//#undef ACTION_6 
//#undef ACTION_7 
//#undef ACTION_8 
//#undef ACTION_9 
//#undef ACTION_10
//#undef ACTION_11
//#undef ACTION_12
//#undef ACTION_13
//#undef ACTION_14
//
//#undef CONDITION_A 
//#undef CONDITION_B 
//#undef CONDITION_N1
//#undef CONDITION_N2
//#undef CONDITION_N3
//#undef CONDITION_N4
//#undef CONDITION_N5
//    }
//    void SecondScan()
//    {
//        n_labels_ = LabelsSolver::Flatten();
//
//        // Second scan
//        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
//            unsigned int *img_labels_row_start = img_labels_.ptr<unsigned int>(r_i);
//            unsigned int *img_labels_row_end = img_labels_row_start + img_labels_.cols;
//            unsigned int *img_labels_row = img_labels_row_start;
//            for (int c_i = 0; img_labels_row != img_labels_row_end; ++img_labels_row, ++c_i) {
//                *img_labels_row = LabelsSolver::GetLabel(*img_labels_row);
//            }
//        }
//    }
};

#endif // YACCLAB_LABELING_HE_2008_H_