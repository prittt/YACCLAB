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

    CircularBuffer() = default;

    CircularBuffer(size_t capacity) : buffer_{ new T[capacity] }, capacity_{ capacity } {}
    
    ~CircularBuffer()
    {
        delete[] buffer_;
    }

    // Basic exception safety
    void Create(size_t capacity) {
        delete[] buffer_;

        capacity_ = 0;
        size_ = 0;
        tail_ = 0;
        head_ = 0;
        buffer_ = nullptr;

        buffer_ = new T[capacity];
        capacity_ = capacity;
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
            unsigned int* const img_labels_row_prev = (unsigned int*)(((char*)img_labels_row) - img_labels_.step.p[0]);

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
            unsigned int* img_labels_row_start = img_labels_.ptr<unsigned int>(r_i);
            unsigned int* img_labels_row_end = img_labels_row_start + img_labels_.cols;
            unsigned int* img_labels_row = img_labels_row_start;
            for (int c_i = 0; img_labels_row != img_labels_row_end; ++img_labels_row, ++c_i) {
                *img_labels_row = LabelsSolver::GetLabel(*img_labels_row);
            }
        }

        LabelsSolver::Dealloc();
    }

    void PerformLabelingWithSteps()
    {
        double alloc_timing = Alloc();

        perf_.start();
        FirstScan();
        perf_.stop();
        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

        perf_.start();
        SecondScan();
        perf_.stop();
        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);

    }

private:
    double Alloc()
    {
        // Memory allocation of the labels solver
        double ls_t = LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY, perf_);
        // Memory allocation for the output image
        perf_.start();
        img_labels_ = cv::Mat1i(img_.size());
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double t = perf_.last();
        perf_.start();
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double ma_t = t - perf_.last();

        perf_.start();
        int w(img_.cols); // w = N in the paper naming convention
        // Circular Buffers
        int buffer_capacity = w / 2 + 2;
        s_queue_.Create(buffer_capacity);
        e_queue_.Create(buffer_capacity);
        memset(s_queue_.buffer_, 0, buffer_capacity);
        memset(e_queue_.buffer_, 0, buffer_capacity);
        perf_.stop();
        double qu_t = perf_.last();
        perf_.start();
        memset(s_queue_.buffer_, 0, buffer_capacity);
        memset(e_queue_.buffer_, 0, buffer_capacity);
        perf_.stop();
        qu_t -= perf_.last();

        // Return total time
        return ls_t + ma_t + qu_t;
    }
    void Dealloc()
    {
        LabelsSolver::Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan()
    {
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart); // Initialization
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

        for (int r = 0; r < h; r += 1) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            // const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int*)(((char*)img_labels_row) - img_labels_.step.p[0]);

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
                int s = s_queue_.Enqueue(w * r + c++); // Current run starting index
                for (; c < w && img_row[c] > 0; c += 1) {}
                int crb = c != w; // Consider right border (derived by c - 1 != w - 1)
                int e = e_queue_.Enqueue(w * r + --c); // Current run end index

                // 2) Discard all the runs until a run r(h,t) that end after/at s - N - 1 (t in the following scheme)
                // +-+-+-+-+-+-+-+-+-+
                // | |t|>| | | | | | |
                // +-+-+-+-+-+-+-+-+-+
                // | | |s| | | |e| | |
                // +-+-+-+-+-+-+-+-+-+
                // This run always exists. It is at most the run we just found.
                int h = s_queue_.Front();
                int t = e_queue_.Front();
                while (t < s - w - clb) {
                    h = s_queue_.DequeueAndFront();
                    t = e_queue_.DequeueAndFront();
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
                        h = s_queue_.DequeueAndFront();
                        t = e_queue_.DequeueAndFront();
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
                        s_queue_.Dequeue();
                        e_queue_.Dequeue();
                        h = s_queue_.Front();
                        t = e_queue_.Front();
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
        
    }
    void SecondScan()
    {
        n_labels_ = LabelsSolver::Flatten();

        // Second scan
        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
            unsigned int* img_labels_row_start = img_labels_.ptr<unsigned int>(r_i);
            unsigned int* img_labels_row_end = img_labels_row_start + img_labels_.cols;
            unsigned int* img_labels_row = img_labels_row_start;
            for (int c_i = 0; img_labels_row != img_labels_row_end; ++img_labels_row, ++c_i) {
                *img_labels_row = LabelsSolver::GetLabel(*img_labels_row);
            }
        }
    }

    CircularBuffer<int> s_queue_;
    CircularBuffer<int> e_queue_;
};

#endif // YACCLAB_LABELING_HE_2008_H_