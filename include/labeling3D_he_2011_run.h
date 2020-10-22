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

#ifndef YACCLAB_LABELING3D_HE_2011_RUN_H_
#define YACCLAB_LABELING3D_HE_2011_RUN_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

struct Run
{
    uint16_t start = 0;
    uint16_t end = 0;
    unsigned label = 0;
};

class Table3D
{
public:

    size_t d_;
    size_t h_;
    size_t max_runs_;
    Run *data_;       // This vector stores run data for each row of each slice
    uint16_t *sizes_; // This vector stores the number of runs actually contained in each row of each slice

    void Setup(size_t d, size_t h, size_t w)
    {
        d_ = d;
        h_ = h;
        max_runs_ = w / 2 + 1;
    }

    double Alloc(PerformanceEvaluator& perf)
    {
        perf.start();
        data_ = new Run[d_ * h_ * max_runs_];
        sizes_ = new uint16_t[d_ * h_];
        memset(data_, 0, d_ * h_ * max_runs_ * sizeof(Run));
        memset(sizes_, 0, d_ * h_ * sizeof(uint16_t));
        perf.stop();
        double t = perf.last();
        perf.start();
        memset(data_, 0, d_ * h_ * max_runs_ * sizeof(Run));
        memset(sizes_, 0, d_ * h_ * sizeof(uint16_t));
        perf.stop();
        return t - perf.last();
    }

    void Alloc()
    {
        data_ = new Run[d_ * h_ * max_runs_];
        sizes_ = new uint16_t[d_ * h_];
    }

    void Dealloc()
    {
        delete[] data_;
        delete[] sizes_;
    }
};

template <typename LabelsSolver>
class RBTS_3D : public Labeling3D<CONN_26>
{
public:
    RBTS_3D() {}

    Table3D runs;

    static inline int ProcessRun(uint16_t row_index, uint16_t row_nruns, Run* row_runs, Run* cur_run, bool *new_label)
    {
        // Discard previous non connected runs (step "2" of the 2D algorithm)
        for (;
            row_index < row_nruns &&
            row_runs[row_index].end < cur_run->start - 1;
            ++row_index) {
        }

        // Get label (step "3A" of the 2D algorithm)
        if (row_index < row_nruns &&
            row_runs[row_index].start <= cur_run->end + 1) {
            if (*new_label) {
                cur_run->label = row_runs[row_index].label;
                *new_label = false;
            }
            else {
                LabelsSolver::Merge(cur_run->label, row_runs[row_index].label);
            }
        }

        // Merge label (step "3B" of the 2D algorithm)
        for (;
            row_index < row_nruns &&
            row_runs[row_index].end <= cur_run->end;
            ++row_index) {
            LabelsSolver::Merge(cur_run->label, row_runs[row_index].label);
        }

        // Get label without "removing the run" (step "4" of the 2D algorithm)
        // the skip step is not required in this case because this algorithm does not employ
        // a circular buffer.
        if (row_index < row_nruns &&
            row_runs[row_index].start <= cur_run->end + 1) {
            LabelsSolver::Merge(cur_run->label, row_runs[row_index].label);
        }
        return row_index;
    }

    void PerformLabeling()
    {
        int d = img_.size.p[0];
        int h = img_.size.p[1];
        int w = img_.size.p[2];

        img_labels_.create(3, img_.size.p, CV_32SC1);
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);

        runs.Setup(d, h, w);
        runs.Alloc();

        LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY); // Memory allocation of the labels solver
        LabelsSolver::Setup(); // Labels solver initialization

        // First scan
        Run* run_slice00_row00 = runs.data_;
        uint16_t* nruns_slice00_row00 = runs.sizes_;
        for (int s = 0; s < d; s++) {

            for (int r = 0; r < h; r++) {
                // Row pointers for the input image
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);

                int slice11_row00_index = 0;
                int slice11_row11_index = 0;
                int slice11_row01_index = 0;
                int slice00_row11_index = 0;

                int nruns = 0;

                Run* run_slice11_row00 = run_slice00_row00 - (runs.max_runs_) * runs.h_;
                Run* run_slice11_row11 = run_slice11_row00 - (runs.max_runs_);
                Run* run_slice11_row01 = run_slice11_row00 + (runs.max_runs_);

                Run* run_slice00_row11 = run_slice00_row00 - (runs.max_runs_);

                uint16_t* nruns_slice11_row00 = nruns_slice00_row00 - h;
                uint16_t* nruns_slice11_row11 = nruns_slice11_row00 - 1;
                uint16_t* nruns_slice11_row01 = nruns_slice11_row00 + 1;

                uint16_t* nruns_slice00_row11 = nruns_slice00_row00 - 1;

                for (int c = 0; c < w; c++) {
                    // Is there a new run ?
                    if (img_slice00_row00[c] == 0) {
                        continue;
                    }

                    // Yes (new run)
                    bool new_label = true;
                    run_slice00_row00[nruns].start = c; // We start from 1 because 0 is a "special" run
                                                        // to store additional info
                    for (; c < w && img_slice00_row00[c] > 0; ++c) {}
                    run_slice00_row00[nruns].end = c - 1;

                    if (s > 0) {
                        if (r > 0) {
                            slice11_row11_index = ProcessRun(slice11_row11_index,        // uint16_t row_index
                                *nruns_slice11_row11,       // uint16_t row_nruns
                                run_slice11_row11,          // Run* row_runs
                                &run_slice00_row00[nruns],  // Run* cur_run
                                &new_label                  // bool *new_label
                            );
                        }
                        slice11_row00_index = ProcessRun(slice11_row00_index,        // uint16_t row_index
                            *nruns_slice11_row00,       // uint16_t row_nruns
                            run_slice11_row00,          // Run* row_runs
                            &run_slice00_row00[nruns],  // Run* cur_run
                            &new_label                  // bool *new_label
                        );
                        if (r < h - 1) {
                            slice11_row01_index = ProcessRun(slice11_row01_index,        // uint16_t row_index
                                *nruns_slice11_row01,       // uint16_t row_nruns
                                run_slice11_row01,          // Run* row_runs
                                &run_slice00_row00[nruns],  // Run* cur_run
                                &new_label                  // bool *new_label
                            );
                        }
                    }

                    if (r > 0) {
                        slice00_row11_index = ProcessRun(slice00_row11_index,        // uint16_t row_index
                            *nruns_slice00_row11,       // uint16_t row_nruns
                            run_slice00_row11,          // Run* row_runs
                            &run_slice00_row00[nruns],  // Run* cur_run
                            &new_label                  // bool *new_label
                        );
                    }

                    if (new_label) {
                        run_slice00_row00[nruns].label = LabelsSolver::NewLabel();
                    }
                    nruns++;
                } // Columns cycle end

                run_slice00_row00 += (runs.max_runs_);
                (*nruns_slice00_row00++) = nruns;
            } // Rows cycle end
        } // Planes cycle end

        // Second scan
        LabelsSolver::Flatten();

        int* img_row = reinterpret_cast<int*>(img_labels_.data);
        Run* run_row = runs.data_;
        uint16_t* nruns = runs.sizes_;
        for (int s = 0; s < d; s++) {
            for (int r = 0; r < h; r++) {
                for (int id = 0; id < *nruns; id++) {
                    for (int c = run_row[id].start; c <= run_row[id].end; ++c) {
                        img_row[c] = LabelsSolver::GetLabel(run_row[id].label);
                    }
                }
                run_row += (runs.max_runs_);
                img_row += img_labels_.step[1] / sizeof(int);
                nruns++;
            }
        }

        LabelsSolver::Dealloc(); // Memory deallocation of the labels solver
        runs.Dealloc(); // Memory deallocation of the Table3D
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
        double ls_t = LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY, perf_);
        // Memory allocation of Table3D 
        runs.Setup(img_.size.p[0], img_.size.p[1], img_.size.p[2]);
        ls_t += runs.Alloc(perf_);
        // Memory allocation for the output image
        perf_.start();
        img_labels_.create(3, img_.size.p, CV_32SC1);
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double t = perf_.last();
        perf_.start();
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double ma_t = t - perf_.last();
        // Return total time
        return ls_t + ma_t;
    }
    void Dealloc()
    {
        LabelsSolver::Dealloc();
        runs.Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan()
    {
        int d = img_.size.p[0];
        int h = img_.size.p[1];
        int w = img_.size.p[2];

        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);

        LabelsSolver::Setup(); // Labels solver initialization
        
        // First scan
        Run* run_slice00_row00 = runs.data_;
        uint16_t* nruns_slice00_row00 = runs.sizes_;
        for (int s = 0; s < d; s++) {

            for (int r = 0; r < h; r++) {
                // Row pointers for the input image
                const unsigned char* const img_slice00_row00 = img_.ptr<unsigned char>(s, r);

                int slice11_row00_index = 0;
                int slice11_row11_index = 0;
                int slice11_row01_index = 0;
                int slice00_row11_index = 0;

                int nruns = 0;

                Run* run_slice11_row00 = run_slice00_row00 - (runs.max_runs_) * runs.h_;
                Run* run_slice11_row11 = run_slice11_row00 - (runs.max_runs_);
                Run* run_slice11_row01 = run_slice11_row00 + (runs.max_runs_);

                Run* run_slice00_row11 = run_slice00_row00 - (runs.max_runs_);

                uint16_t* nruns_slice11_row00 = nruns_slice00_row00 - h;
                uint16_t* nruns_slice11_row11 = nruns_slice11_row00 - 1;
                uint16_t* nruns_slice11_row01 = nruns_slice11_row00 + 1;

                uint16_t* nruns_slice00_row11 = nruns_slice00_row00 - 1;

                for (int c = 0; c < w; c++) {
                    // Is there a new run ?
                    if (img_slice00_row00[c] == 0) {
                        continue;
                    }

                    // Yes (new run)
                    bool new_label = true;
                    run_slice00_row00[nruns].start = c; // We start from 1 because 0 is a "special" run
                                                        // to store additional info
                    for (; c < w && img_slice00_row00[c] > 0; ++c) {}
                    run_slice00_row00[nruns].end = c - 1;

                    if (s > 0) {
                        if (r > 0) {
                            slice11_row11_index = ProcessRun(slice11_row11_index,        // uint16_t row_index
                                                             *nruns_slice11_row11,       // uint16_t row_nruns
                                                             run_slice11_row11,          // Run* row_runs
                                                             &run_slice00_row00[nruns],  // Run* cur_run
                                                             &new_label                  // bool *new_label
                                                            );
                        }
                        slice11_row00_index = ProcessRun(slice11_row00_index,        // uint16_t row_index
                                                         *nruns_slice11_row00,       // uint16_t row_nruns
                                                         run_slice11_row00,          // Run* row_runs
                                                         &run_slice00_row00[nruns],  // Run* cur_run
                                                         &new_label                  // bool *new_label
                                                        );
                        if (r < h - 1) {
                            slice11_row01_index = ProcessRun(slice11_row01_index,        // uint16_t row_index
                                                             *nruns_slice11_row01,       // uint16_t row_nruns
                                                             run_slice11_row01,          // Run* row_runs
                                                             &run_slice00_row00[nruns],  // Run* cur_run
                                                             &new_label                  // bool *new_label
                                                            );
                        }
                    }

                    if (r > 0) {
                        slice00_row11_index = ProcessRun(slice00_row11_index,        // uint16_t row_index
                                                         *nruns_slice00_row11,       // uint16_t row_nruns
                                                         run_slice00_row11,          // Run* row_runs
                                                         &run_slice00_row00[nruns],  // Run* cur_run
                                                         &new_label                  // bool *new_label
                                                        );
                    }

                    if (new_label) {
                        run_slice00_row00[nruns].label = LabelsSolver::NewLabel();
                    }
                    nruns++;
                } // Columns cycle end

                run_slice00_row00 += (runs.max_runs_);
                (*nruns_slice00_row00++) = nruns;
            } // Rows cycle end
        } // Planes cycle end
    }
    void SecondScan()
    {
        int d = img_.size.p[0];
        int h = img_.size.p[1];
        // int w = img_.size.p[2];

        // Second scan
        LabelsSolver::Flatten();

        int* img_row = reinterpret_cast<int*>(img_labels_.data);
        Run* run_row = runs.data_;
        uint16_t* nruns = runs.sizes_;
        for (int s = 0; s < d; s++) {
            for (int r = 0; r < h; r++) {
                for (int id = 0; id < *nruns; id++) {
                    for (int c = run_row[id].start; c <= run_row[id].end; ++c) {
                        img_row[c] = LabelsSolver::GetLabel(run_row[id].label);
                    }
                }
                run_row += (runs.max_runs_);
                img_row += img_labels_.step[1] / sizeof(int);
                nruns++;
            }
        }
    }
};

#endif // YACCLAB_LABELING3D_HE_2011_RUN_H_