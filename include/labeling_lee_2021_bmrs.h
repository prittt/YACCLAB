// Copyright (c) 2021, the YACCLAB contributors, as 
// shown by the AUTHORS file, plus additional authors
// listed below. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional Authors:
// Wonsang Lee
// Department of Physics
// Konkuk University, Korea


#ifndef YACCLAB_LABELING_BMRS
#define YACCLAB_LABELING_BMRS
#include <vector>
#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"
#include "bit_scan_forward.h"


template <typename LabelsSolver>
class BMRS : public Labeling2D<Connectivity2D::CONN_8>
{
    struct Data_Compressed {
        uint64_t* bits;
        int height;
        int width;
        int data_width;
        uint64_t* operator [](uint64_t row) {
            return bits + data_width * row;
        }
        void Alloc(int _height, int _width) {
            height = _height, width = _width;
            data_width = _width / 64 + 1;
            bits = new uint64_t[height * data_width];
        }
        void Dealloc() {
            delete[] bits;
        }
    };
    struct Run {
        unsigned short start_pos;
        unsigned short end_pos;
        unsigned label;
    };
    struct Runs {
        Run* runs;
        unsigned height;
        unsigned  width;
        void Alloc(int _height, int _width) {
            height = _height, width = _width;
            runs = new Run[height * (width / 2 + 2) + 1];
        }
        void Dealloc() {
            delete[] runs;
        }
    };
    Data_Compressed data_compressed;
    Data_Compressed data_merged;
    Data_Compressed data_flags;
    Runs data_runs;

public:
    BMRS() {}
    void PerformLabeling()
    {
        int w(img_.cols);
        int h(img_.rows);

        data_compressed.Alloc(h, w);
        InitCompressedData(data_compressed);

        //generate merged data
        int h_merge = h / 2 + h % 2;
        int data_width = data_compressed.data_width;
        data_merged.Alloc(h_merge, w);
        for (int i = 0; i < h / 2; i++) {
            uint64_t* pdata_source1 = data_compressed[2 * i];
            uint64_t* pdata_source2 = data_compressed[2 * i + 1];
            uint64_t* pdata_merged = data_merged[i];
            for (int j = 0; j < data_width; j++) pdata_merged[j] = pdata_source1[j] | pdata_source2[j];
        }
        if (h % 2) {
            uint64_t* pdata_source = data_compressed[h - 1];
            uint64_t* pdata_merged = data_merged[h / 2];
            for (int j = 0; j < data_width; j++) pdata_merged[j] = pdata_source[j];
        }

        //generate flag bits
        data_flags.Alloc(h_merge - 1, w);
        for (int i = 0; i < data_flags.height; i++) {
            uint64_t* bits_u = data_compressed[2 * i + 1];
            uint64_t* bits_d = data_compressed[2 * i + 2];
            uint64_t* bits_dest = data_flags[i];

            uint64_t u0 = bits_u[0];
            uint64_t d0 = bits_d[0];
            bits_dest[0] = (u0 | (u0 << 1)) & (d0 | (d0 << 1));
            for (int j = 1; j < data_width; j++) {
                uint64_t u = bits_u[j];
                uint64_t u_shl = u << 1;
                uint64_t d = bits_d[j];
                uint64_t d_shl = d << 1;
                if (bits_u[j - 1] & 0x8000000000000000) u_shl |= 1;
                if (bits_d[j - 1] & 0x8000000000000000) d_shl |= 1;
                bits_dest[j] = (u | u_shl) & (d | d_shl);
            }
        }

        //find runs
        data_runs.Alloc(h_merge, w);
        LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::Setup();
        FindRuns(data_merged.bits, data_flags.bits, h_merge, data_width, data_runs.runs);

        img_labels_ = cv::Mat1i(img_.size(), 0);    // New version (0-init)

        // New version (only uses bitonal input)
        Run* runs = data_runs.runs;
        for (int r = 0; r < h / 2; r++) {
            const uint64_t* const data_u = data_compressed.bits + data_compressed.data_width * 2 * r;
            const uint64_t* const data_d = data_u + data_compressed.data_width;
            unsigned* const labels_u = img_labels_.ptr<unsigned>(2 * r);
            unsigned* const labels_d = img_labels_.ptr<unsigned>(2 * r + 1);

            uint64_t curr_word_u;
            uint64_t curr_word_d;
            unsigned short curr_word_ind = -1;

            for (int j = 0;; runs++) {
                unsigned short start_pos = runs->start_pos;
                if (start_pos == 0xFFFF) {
                    runs++;
                    break;
                }
                unsigned short end_pos = runs->end_pos;
                int label = LabelsSolver::GetLabel(runs->label);

                if (label) {

                    for (unsigned short c = start_pos; c < end_pos; ++c) {

                        unsigned short needed_word_ind = c / 64;
                        if (needed_word_ind != curr_word_ind) {
                            curr_word_u = data_u[needed_word_ind];
                            curr_word_d = data_d[needed_word_ind];
                            curr_word_ind = needed_word_ind;
                        }

                        uint8_t bit_pos = c % 64;
                        uint64_t mask = 1ull << bit_pos;
                        uint64_t bit_u = curr_word_u & mask;
                        uint64_t bit_d = curr_word_d & mask;

                        if (bit_u) {
                            labels_u[c] = label;
                        }
                        if (bit_d) {
                            labels_d[c] = label;
                        }
                        
                    }

                }
            }
        }
        if (h % 2) {
            unsigned* const labels = img_labels_.ptr<unsigned>(h - 1);
            for (int j = 0;; runs++) {
                unsigned short start_pos = runs->start_pos;
                if (start_pos == 0xFFFF) {
                    break;
                }
                unsigned short end_pos = runs->end_pos;
                int label = LabelsSolver::GetLabel(runs->label);
                for (j = start_pos; j < end_pos; j++) {
                    labels[j] = label;
                }
            }
        }


        LabelsSolver::Dealloc();
        data_runs.Dealloc();
        data_flags.Dealloc();
        data_merged.Dealloc();
        data_compressed.Dealloc();
    }
    void PerformLabelingWithSteps() {
        double alloc_timing = Alloc();
        InitCompressedData(data_compressed);

        perf_.start();
        FirstScan();
        perf_.stop();
        perf_.store(Step(StepType::FIRST_SCAN), perf_.last());

        perf_.start();
        SecondScan();
        perf_.stop();
        perf_.store(Step(StepType::SECOND_SCAN), perf_.last());

        double t_runalloc = Get_RunAllocTime();
        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing + t_runalloc);
    }
    void PerformLabelingMem(std::vector<uint64_t>& accesses) {
        //Data structure for memory test
        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0);

        int w(img_.cols);
        int h(img_.rows);

        data_compressed.Alloc(h, w);
        InitCompressedDataMem(data_compressed, img);

        //generate merged data
        int h_merge = h / 2 + h % 2;
        int data_width = data_compressed.data_width;
        data_merged.Alloc(h_merge, w);
        for (int i = 0; i < h / 2; i++) {
            uint64_t* pdata_source1 = data_compressed[2 * i];
            uint64_t* pdata_source2 = data_compressed[2 * i + 1];
            uint64_t* pdata_merged = data_merged[i];
            for (int j = 0; j < data_width; j++) pdata_merged[j] = pdata_source1[j] | pdata_source2[j];
        }
        if (h % 2) {
            uint64_t* pdata_source = data_compressed[h - 1];
            uint64_t* pdata_merged = data_merged[h / 2];
            for (int j = 0; j < data_width; j++) pdata_merged[j] = pdata_source[j];
        }

        //generate flag bits
        data_flags.Alloc(h_merge - 1, w);
        for (int i = 0; i < data_flags.height; i++) {
            uint64_t* bits_u = data_compressed[2 * i + 1];
            uint64_t* bits_d = data_compressed[2 * i + 2];
            uint64_t* bits_dest = data_flags[i];

            uint64_t u0 = bits_u[0];
            uint64_t d0 = bits_d[0];
            bits_dest[0] = (u0 | (u0 << 1)) & (d0 | (d0 << 1));
            for (int j = 1; j < data_width; j++) {
                uint64_t u = bits_u[j];
                uint64_t u_shl = u << 1;
                uint64_t d = bits_d[j];
                uint64_t d_shl = d << 1;
                if (bits_u[j - 1] & 0x8000000000000000) u_shl |= 1;
                if (bits_d[j - 1] & 0x8000000000000000) d_shl |= 1;
                bits_dest[j] = (u | u_shl) & (d | d_shl);
            }
        }

        //find runs
        data_runs.Alloc(h, w);
        LabelsSolver::MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
        LabelsSolver::MemSetup();
        FindRunsMem(data_merged.bits, data_flags.bits, h_merge, data_width, data_runs.runs);

        Run* runs = data_runs.runs;
        for (int i = 0; i < h / 2; i++) {
            int i_u = 2 * i;
            int i_d = i_u + 1;

            for (int j = 0;; runs++) {
                unsigned short start_pos = runs->start_pos;
                if (start_pos == 0xFFFF) {
                    for (int k = j; k < w; k++) img_labels(i_u, k) = 0;
                    for (int k = j; k < w; k++) img_labels(i_d, k) = 0;
                    runs++;
                    break;
                }
                unsigned short end_pos = runs->end_pos;
                int label = LabelsSolver::MemGetLabel(runs->label);

                for (; j < start_pos; j++) img_labels(i_u, j) = 0, img_labels(i_d, j) = 0;
                for (; j < end_pos; j++) {
                    img_labels(i_u, j) = img(i_u, j) ? label : 0;
                    img_labels(i_d, j) = img(i_d, j) ? label : 0;
                }
            }
        }
        if (h % 2) {
            int i = h - 1;
            for (int j = 0;; runs++) {
                unsigned short start_pos = runs->start_pos;
                if (start_pos == 0xFFFF) {
                    for (int k = j; k < w; k++) img_labels(i, j) = 0;
                    break;
                }
                unsigned short end_pos = runs->end_pos;
                int label = LabelsSolver::MemGetLabel(runs->label);
                for (; j < start_pos; j++) img_labels(i, j) = 0;
                for (j = start_pos; j < end_pos; j++) img_labels(i, j) = label;
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (uint64_t)LabelsSolver::MemTotalAccesses();

        img_labels_ = img_labels.GetImage();

        LabelsSolver::MemDealloc();
        data_runs.Dealloc();
        data_flags.Dealloc();
        data_merged.Dealloc();
        data_compressed.Dealloc();
    }

private:
    void InitCompressedData(Data_Compressed& data_compressed) {
        int w(img_.cols);
        int h(img_.rows);

        for (int i = 0; i < h; i++) {
            uint64_t* mbits = data_compressed[i];
            uchar* source = img_.ptr<uchar>(i);
            for (int j = 0; j < w >> 6; j++) {
                uchar* base = source + (j << 6);
                uint64_t obits = 0;
                if (base[0]) obits |= 0x0000000000000001;
                if (base[1]) obits |= 0x0000000000000002;
                if (base[2]) obits |= 0x0000000000000004;
                if (base[3]) obits |= 0x0000000000000008;
                if (base[4]) obits |= 0x0000000000000010;
                if (base[5]) obits |= 0x0000000000000020;
                if (base[6]) obits |= 0x0000000000000040;
                if (base[7]) obits |= 0x0000000000000080;
                if (base[8]) obits |= 0x0000000000000100;
                if (base[9]) obits |= 0x0000000000000200;
                if (base[10]) obits |= 0x0000000000000400;
                if (base[11]) obits |= 0x0000000000000800;
                if (base[12]) obits |= 0x0000000000001000;
                if (base[13]) obits |= 0x0000000000002000;
                if (base[14]) obits |= 0x0000000000004000;
                if (base[15]) obits |= 0x0000000000008000;
                if (base[16]) obits |= 0x0000000000010000;
                if (base[17]) obits |= 0x0000000000020000;
                if (base[18]) obits |= 0x0000000000040000;
                if (base[19]) obits |= 0x0000000000080000;
                if (base[20]) obits |= 0x0000000000100000;
                if (base[21]) obits |= 0x0000000000200000;
                if (base[22]) obits |= 0x0000000000400000;
                if (base[23]) obits |= 0x0000000000800000;
                if (base[24]) obits |= 0x0000000001000000;
                if (base[25]) obits |= 0x0000000002000000;
                if (base[26]) obits |= 0x0000000004000000;
                if (base[27]) obits |= 0x0000000008000000;
                if (base[28]) obits |= 0x0000000010000000;
                if (base[29]) obits |= 0x0000000020000000;
                if (base[30]) obits |= 0x0000000040000000;
                if (base[31]) obits |= 0x0000000080000000;
                if (base[32]) obits |= 0x0000000100000000;
                if (base[33]) obits |= 0x0000000200000000;
                if (base[34]) obits |= 0x0000000400000000;
                if (base[35]) obits |= 0x0000000800000000;
                if (base[36]) obits |= 0x0000001000000000;
                if (base[37]) obits |= 0x0000002000000000;
                if (base[38]) obits |= 0x0000004000000000;
                if (base[39]) obits |= 0x0000008000000000;
                if (base[40]) obits |= 0x0000010000000000;
                if (base[41]) obits |= 0x0000020000000000;
                if (base[42]) obits |= 0x0000040000000000;
                if (base[43]) obits |= 0x0000080000000000;
                if (base[44]) obits |= 0x0000100000000000;
                if (base[45]) obits |= 0x0000200000000000;
                if (base[46]) obits |= 0x0000400000000000;
                if (base[47]) obits |= 0x0000800000000000;
                if (base[48]) obits |= 0x0001000000000000;
                if (base[49]) obits |= 0x0002000000000000;
                if (base[50]) obits |= 0x0004000000000000;
                if (base[51]) obits |= 0x0008000000000000;
                if (base[52]) obits |= 0x0010000000000000;
                if (base[53]) obits |= 0x0020000000000000;
                if (base[54]) obits |= 0x0040000000000000;
                if (base[55]) obits |= 0x0080000000000000;
                if (base[56]) obits |= 0x0100000000000000;
                if (base[57]) obits |= 0x0200000000000000;
                if (base[58]) obits |= 0x0400000000000000;
                if (base[59]) obits |= 0x0800000000000000;
                if (base[60]) obits |= 0x1000000000000000;
                if (base[61]) obits |= 0x2000000000000000;
                if (base[62]) obits |= 0x4000000000000000;
                if (base[63]) obits |= 0x8000000000000000;
                *mbits = obits, mbits++;
            }
            uint64_t obits_final = 0;
            int jbase = w - (w % 64);
            for (int j = 0; j < w % 64; j++) {
                if (source[jbase + j]) obits_final |= ((uint64_t)1 << j);
            }
            *mbits = obits_final;
        }
    }
    void InitCompressedDataMem(Data_Compressed& data_compressed, MemMat<unsigned char>& img) {
        int w(img_.cols);
        int h(img_.rows);

        for (int i = 0; i < h; i++) {
            uint64_t* mbits = data_compressed[i];
            uchar* source = img_.ptr<uchar>(i);
            for (int j = 0; j < w >> 6; j++) {
                int j_base = (j << 6);
                uint64_t obits = 0;
                if (img(i, j_base + 0)) obits |= 0x0000000000000001;
                if (img(i, j_base + 1)) obits |= 0x0000000000000002;
                if (img(i, j_base + 2)) obits |= 0x0000000000000004;
                if (img(i, j_base + 3)) obits |= 0x0000000000000008;
                if (img(i, j_base + 4)) obits |= 0x0000000000000010;
                if (img(i, j_base + 5)) obits |= 0x0000000000000020;
                if (img(i, j_base + 6)) obits |= 0x0000000000000040;
                if (img(i, j_base + 7)) obits |= 0x0000000000000080;
                if (img(i, j_base + 8)) obits |= 0x0000000000000100;
                if (img(i, j_base + 9)) obits |= 0x0000000000000200;
                if (img(i, j_base + 10)) obits |= 0x0000000000000400;
                if (img(i, j_base + 11)) obits |= 0x0000000000000800;
                if (img(i, j_base + 12)) obits |= 0x0000000000001000;
                if (img(i, j_base + 13)) obits |= 0x0000000000002000;
                if (img(i, j_base + 14)) obits |= 0x0000000000004000;
                if (img(i, j_base + 15)) obits |= 0x0000000000008000;
                if (img(i, j_base + 16)) obits |= 0x0000000000010000;
                if (img(i, j_base + 17)) obits |= 0x0000000000020000;
                if (img(i, j_base + 18)) obits |= 0x0000000000040000;
                if (img(i, j_base + 19)) obits |= 0x0000000000080000;
                if (img(i, j_base + 20)) obits |= 0x0000000000100000;
                if (img(i, j_base + 21)) obits |= 0x0000000000200000;
                if (img(i, j_base + 22)) obits |= 0x0000000000400000;
                if (img(i, j_base + 23)) obits |= 0x0000000000800000;
                if (img(i, j_base + 24)) obits |= 0x0000000001000000;
                if (img(i, j_base + 25)) obits |= 0x0000000002000000;
                if (img(i, j_base + 26)) obits |= 0x0000000004000000;
                if (img(i, j_base + 27)) obits |= 0x0000000008000000;
                if (img(i, j_base + 28)) obits |= 0x0000000010000000;
                if (img(i, j_base + 29)) obits |= 0x0000000020000000;
                if (img(i, j_base + 30)) obits |= 0x0000000040000000;
                if (img(i, j_base + 31)) obits |= 0x0000000080000000;
                if (img(i, j_base + 32)) obits |= 0x0000000100000000;
                if (img(i, j_base + 33)) obits |= 0x0000000200000000;
                if (img(i, j_base + 34)) obits |= 0x0000000400000000;
                if (img(i, j_base + 35)) obits |= 0x0000000800000000;
                if (img(i, j_base + 36)) obits |= 0x0000001000000000;
                if (img(i, j_base + 37)) obits |= 0x0000002000000000;
                if (img(i, j_base + 38)) obits |= 0x0000004000000000;
                if (img(i, j_base + 39)) obits |= 0x0000008000000000;
                if (img(i, j_base + 40)) obits |= 0x0000010000000000;
                if (img(i, j_base + 41)) obits |= 0x0000020000000000;
                if (img(i, j_base + 42)) obits |= 0x0000040000000000;
                if (img(i, j_base + 43)) obits |= 0x0000080000000000;
                if (img(i, j_base + 44)) obits |= 0x0000100000000000;
                if (img(i, j_base + 45)) obits |= 0x0000200000000000;
                if (img(i, j_base + 46)) obits |= 0x0000400000000000;
                if (img(i, j_base + 47)) obits |= 0x0000800000000000;
                if (img(i, j_base + 48)) obits |= 0x0001000000000000;
                if (img(i, j_base + 49)) obits |= 0x0002000000000000;
                if (img(i, j_base + 50)) obits |= 0x0004000000000000;
                if (img(i, j_base + 51)) obits |= 0x0008000000000000;
                if (img(i, j_base + 52)) obits |= 0x0010000000000000;
                if (img(i, j_base + 53)) obits |= 0x0020000000000000;
                if (img(i, j_base + 54)) obits |= 0x0040000000000000;
                if (img(i, j_base + 55)) obits |= 0x0080000000000000;
                if (img(i, j_base + 56)) obits |= 0x0100000000000000;
                if (img(i, j_base + 57)) obits |= 0x0200000000000000;
                if (img(i, j_base + 58)) obits |= 0x0400000000000000;
                if (img(i, j_base + 59)) obits |= 0x0800000000000000;
                if (img(i, j_base + 60)) obits |= 0x1000000000000000;
                if (img(i, j_base + 61)) obits |= 0x2000000000000000;
                if (img(i, j_base + 62)) obits |= 0x4000000000000000;
                if (img(i, j_base + 63)) obits |= 0x8000000000000000;
                *mbits = obits, mbits++;
            }
            uint64_t obits_final = 0;
            int jbase = w - (w % 64);
            for (int j = 0; j < w % 64; j++) {
                if (img(i, jbase + j)) obits_final |= ((uint64_t)1 << j);
            }
            *mbits = obits_final;
        }
    }
    void FindRuns(const uint64_t* bits_start, const uint64_t* bits_flag, int height, int data_width, Run* runs) {
        Run* runs_up = runs;

        //process runs in the first merged row
        const uint64_t* bits = bits_start;
        const uint64_t* bit_final = bits + data_width;
        uint64_t working_bits = *bits;
        unsigned long basepos = 0, bitpos = 0;
        for (;; runs++) {
            //find starting position
            while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                bits++, basepos += 64;
                if (bits == bit_final) {
                    runs->start_pos = (short)0xFFFF;
                    runs->end_pos = (short)0xFFFF;
                    runs++;
                    goto out;
                }
                working_bits = *bits;
            }
            runs->start_pos = short(basepos + bitpos);

            //find ending position
            working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
            while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                bits++, basepos += 64;
                working_bits = ~(*bits);
            }
            working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
            runs->end_pos = short(basepos + bitpos);
            runs->label = LabelsSolver::NewLabel();
        }
    out:

        //process runs in the rests
        for (int row = 1; row < height; row++) {
            Run* runs_save = runs;
            const uint64_t* bits_f = bits_flag + data_width * (row - 1);
            const uint64_t* bits = bits_start + data_width * row;
            const uint64_t* bit_final = bits + data_width;
            uint64_t working_bits = *bits;
            unsigned long basepos = 0, bitpos = 0;

            for (;; runs++) {
                //find starting position
                while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                    bits++, basepos += 64;
                    if (bits == bit_final) {
                        runs->start_pos = (short)0xFFFF;
                        runs->end_pos = (short)0xFFFF;
                        runs++;
                        goto out2;
                    }
                    working_bits = *bits;
                }
                unsigned short start_pos = short(basepos + bitpos);

                //find ending position
                working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
                while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                    bits++, basepos += 64;
                    working_bits = ~(*bits);
                }
                working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
                unsigned short end_pos = short(basepos + bitpos);

                //Skip upper runs end before this slice starts
                for (; runs_up->end_pos < start_pos; runs_up++);

                //No upper run meets this
                if (runs_up->start_pos > end_pos) {
                    runs->start_pos = start_pos;
                    runs->end_pos = end_pos;
                    runs->label = LabelsSolver::NewLabel();
                    continue;
                };

                //Next upper run can not meet this
                unsigned short cross_st = (start_pos >= runs_up->start_pos) ? start_pos : runs_up->start_pos;
                if (end_pos <= runs_up->end_pos) {
                    if (is_connected(bits_f, cross_st, end_pos)) runs->label = LabelsSolver::GetLabel(runs_up->label);
                    else runs->label = LabelsSolver::NewLabel();
                    runs->start_pos = start_pos;
                    runs->end_pos = end_pos;
                    continue;
                }

                unsigned label;
                if (is_connected(bits_f, cross_st, runs_up->end_pos)) label = LabelsSolver::GetLabel(runs_up->label);
                else label = 0;
                runs_up++;

                //Find next upper runs meet this
                for (; runs_up->start_pos <= end_pos; runs_up++) {
                    if (end_pos <= runs_up->end_pos) {
                        if (is_connected(bits_f, runs_up->start_pos, end_pos)) {
                            unsigned label_other = LabelsSolver::GetLabel(runs_up->label);
                            if (label != label_other) {
                                label = (label) ? LabelsSolver::Merge(label, label_other) : label_other;
                            }
                        }
                        break;
                    }
                    else {
                        if (is_connected(bits_f, runs_up->start_pos, runs_up->end_pos)) {
                            unsigned label_other = LabelsSolver::GetLabel(runs_up->label);
                            if (label != label_other) {
                                label = (label) ? LabelsSolver::Merge(label, label_other) : label_other;
                            }
                        }
                    }
                }

                if (label) runs->label = label;
                else runs->label = LabelsSolver::NewLabel();
                runs->start_pos = start_pos;
                runs->end_pos = end_pos;
            }
        out2:
            runs_up = runs_save;
        }
        n_labels_ = LabelsSolver::Flatten();
    }
    void FindRunsMem(const uint64_t* bits_start, const uint64_t* bits_flag, int height, int data_width, Run* runs) {
        Run* runs_up = runs;

        //process runs in the first merged row
        const uint64_t* bits = bits_start;
        const uint64_t* bit_final = bits + data_width;
        uint64_t working_bits = *bits;
        unsigned long basepos = 0, bitpos = 0;
        for (;; runs++) {
            //find starting position
            while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                bits++, basepos += 64;
                if (bits == bit_final) {
                    runs->start_pos = (short)0xFFFF;
                    runs->end_pos = (short)0xFFFF;
                    runs++;
                    goto out;
                }
                working_bits = *bits;
            }
            runs->start_pos = short(basepos + bitpos);

            //find ending position
            working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
            while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                bits++, basepos += 64;
                working_bits = ~(*bits);
            }
            working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
            runs->end_pos = short(basepos + bitpos);
            runs->label = LabelsSolver::MemNewLabel();
        }
    out:

        //process runs in the rests
        for (int row = 1; row < height; row++) {
            Run* runs_save = runs;
            const uint64_t* bits_f = bits_flag + data_width * (row - 1);
            const uint64_t* bits = bits_start + data_width * row;
            const uint64_t* bit_final = bits + data_width;
            uint64_t working_bits = *bits;
            unsigned long basepos = 0, bitpos = 0;

            for (;; runs++) {
                //find starting position
                while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                    bits++, basepos += 64;
                    if (bits == bit_final) {
                        runs->start_pos = (short)0xFFFF;
                        runs->end_pos = (short)0xFFFF;
                        runs++;
                        goto out2;
                    }
                    working_bits = *bits;
                }
                unsigned short start_pos = short(basepos + bitpos);

                //find ending position
                working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
                while (!YacclabBitScanForward64(&bitpos, working_bits)) {
                    bits++, basepos += 64;
                    working_bits = ~(*bits);
                }
                working_bits = (~working_bits) & (0xFFFFFFFFFFFFFFFF << bitpos);
                unsigned short end_pos = short(basepos + bitpos);

                //Skip upper runs end before this slice starts
                for (; runs_up->end_pos < start_pos; runs_up++);

                //No upper run meets this
                if (runs_up->start_pos > end_pos) {
                    runs->start_pos = start_pos;
                    runs->end_pos = end_pos;
                    runs->label = LabelsSolver::MemNewLabel();
                    continue;
                };

                //Next upper run can not meet this
                unsigned short cross_st = (start_pos >= runs_up->start_pos) ? start_pos : runs_up->start_pos;
                if (end_pos <= runs_up->end_pos) {
                    if (is_connected(bits_f, cross_st, end_pos)) runs->label = LabelsSolver::MemGetLabel(runs_up->label);
                    else runs->label = LabelsSolver::MemNewLabel();
                    runs->start_pos = start_pos;
                    runs->end_pos = end_pos;
                    continue;
                }

                unsigned label;
                if (is_connected(bits_f, cross_st, runs_up->end_pos)) label = LabelsSolver::MemGetLabel(runs_up->label);
                else label = 0;
                runs_up++;

                //Find next upper runs meet this
                for (; runs_up->start_pos <= end_pos; runs_up++) {
                    if (end_pos <= runs_up->end_pos) {
                        if (is_connected(bits_f, runs_up->start_pos, end_pos)) {
                            unsigned label_other = LabelsSolver::MemGetLabel(runs_up->label);
                            if (label != label_other) {
                                label = (label) ? LabelsSolver::MemMerge(label, label_other) : label_other;
                            }
                        }
                        break;
                    }
                    else {
                        if (is_connected(bits_f, runs_up->start_pos, runs_up->end_pos)) {
                            unsigned label_other = LabelsSolver::MemGetLabel(runs_up->label);
                            if (label != label_other) {
                                label = (label) ? LabelsSolver::MemMerge(label, label_other) : label_other;
                            }
                        }
                    }
                }

                if (label) runs->label = label;
                else runs->label = LabelsSolver::MemNewLabel();
                runs->start_pos = start_pos;
                runs->end_pos = end_pos;
            }
        out2:
            runs_up = runs_save;
        }
        n_labels_ = LabelsSolver::MemFlatten();
    }
    uint64_t is_connected(const uint64_t* flag_bits, unsigned start, unsigned end) {
        if (start == end) return flag_bits[start >> 6] & ((uint64_t)1 << (start & 0x0000003F));

        unsigned st_base = start >> 6;
        unsigned st_bits = start & 0x0000003F;
        unsigned ed_base = (end + 1) >> 6;
        unsigned ed_bits = (end + 1) & 0x0000003F;
        if (st_base == ed_base) {
            uint64_t cutter = (0xFFFFFFFFFFFFFFFF << st_bits) ^ (0xFFFFFFFFFFFFFFFF << ed_bits);
            return flag_bits[st_base] & cutter;
        }

        for (unsigned i = st_base + 1; i < ed_base; i++) {
            if (flag_bits[i]) return true;
        }
        uint64_t cutter_st = 0xFFFFFFFFFFFFFFFF << st_bits;
        uint64_t cutter_ed = ~(0xFFFFFFFFFFFFFFFF << ed_bits);
        if (flag_bits[st_base] & cutter_st) return true;
        if (flag_bits[ed_base] & cutter_ed) return true;
        return false;
    }

private:
    double Alloc() {
        // Memory allocation of the labels solver
        double ls_t = LabelsSolver::Alloc(UPPER_BOUND_8_CONNECTIVITY, perf_);

        int w(img_.cols);
        int h(img_.rows);
        int h_merge = h / 2 + h % 2;

        // Memory allocation for others
        perf_.start();
        img_labels_ = cv::Mat1i(img_.size());
        data_compressed.Alloc(h, w);
        data_merged.Alloc(h_merge, w);
        data_flags.Alloc(h_merge - 1, w);
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        memset(data_compressed.bits, 0, data_compressed.height * data_compressed.data_width * sizeof(uint64_t));
        memset(data_merged.bits, 0, data_merged.height * data_merged.data_width * sizeof(uint64_t));
        memset(data_flags.bits, 0, data_flags.height * data_flags.data_width * sizeof(uint64_t));
        perf_.stop();
        double t = perf_.last();

        // Run metadata allocation time will be calculated later
        data_runs.Alloc(h_merge, w);

        perf_.start();
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        memset(data_compressed.bits, 0, data_compressed.height * data_compressed.data_width * sizeof(uint64_t));
        memset(data_merged.bits, 0, data_merged.height * data_merged.data_width * sizeof(uint64_t));
        memset(data_flags.bits, 0, data_flags.height * data_flags.data_width * sizeof(uint64_t));
        perf_.stop();
        double ma_t = t - perf_.last();

        // Return total time
        return ls_t + ma_t;
    }
    double Get_RunAllocTime() {
        Run* run_start = data_runs.runs;
        Run* run_end = run_start;
        for (size_t i = 0; i < data_runs.height; i++, run_end++) {
            for (; run_end->start_pos != 0xFFFF; run_end++);
        }
        size_t size_used = (size_t)run_end - (size_t)run_start;

        Runs runs_temp;
        perf_.start();
        runs_temp.Alloc(data_runs.height, data_runs.width);
        memset(runs_temp.runs, 0, size_used);
        perf_.stop();
        double t = perf_.last();

        perf_.start();
        memset(runs_temp.runs, 0, size_used);
        perf_.stop();
        double t_result = t - perf_.last();

        runs_temp.Dealloc();
        return t_result;
    }
    void Dealloc() {
        LabelsSolver::Dealloc();
        data_runs.Dealloc();
        data_flags.Dealloc();
        data_merged.Dealloc();
        data_compressed.Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan() {
        int w(img_.cols);
        int h(img_.rows);

        //generate merged data
        int h_merge = h / 2 + h % 2;
        int data_width = data_compressed.data_width;
        for (int i = 0; i < h / 2; i++) {
            uint64_t* pdata_source1 = data_compressed[2 * i];
            uint64_t* pdata_source2 = data_compressed[2 * i + 1];
            uint64_t* pdata_merged = data_merged[i];
            for (int j = 0; j < data_width; j++) pdata_merged[j] = pdata_source1[j] | pdata_source2[j];
        }
        if (h % 2) {
            uint64_t* pdata_source = data_compressed[h - 1];
            uint64_t* pdata_merged = data_merged[h / 2];
            for (int j = 0; j < data_width; j++) pdata_merged[j] = pdata_source[j];
        }

        //generate flag bits
        for (int i = 0; i < data_flags.height; i++) {
            uint64_t* bits_u = data_compressed[2 * i + 1];
            uint64_t* bits_d = data_compressed[2 * i + 2];
            uint64_t* bits_dest = data_flags[i];

            uint64_t u0 = bits_u[0];
            uint64_t d0 = bits_d[0];
            bits_dest[0] = (u0 | (u0 << 1)) & (d0 | (d0 << 1));
            for (int j = 1; j < data_width; j++) {
                uint64_t u = bits_u[j];
                uint64_t u_shl = u << 1;
                uint64_t d = bits_d[j];
                uint64_t d_shl = d << 1;
                if (bits_u[j - 1] & 0x8000000000000000) u_shl |= 1;
                if (bits_d[j - 1] & 0x8000000000000000) d_shl |= 1;
                bits_dest[j] = (u | u_shl) & (d | d_shl);
            }
        }

        LabelsSolver::Setup();
        FindRuns(data_merged.bits, data_flags.bits, h_merge, data_width, data_runs.runs);
    }
    void SecondScan() {
        int w(img_.cols);
        int h(img_.rows);

        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);

        Run* runs = data_runs.runs;
        for (int r = 0; r < h / 2; r++) {
            const uint64_t* const data_u = data_compressed.bits + data_compressed.data_width * 2 * r;
            const uint64_t* const data_d = data_u + data_compressed.data_width;
            unsigned* const labels_u = img_labels_.ptr<unsigned>(2 * r);
            unsigned* const labels_d = img_labels_.ptr<unsigned>(2 * r + 1);

            uint64_t curr_word_u;
            uint64_t curr_word_d;
            unsigned short curr_word_ind = -1;

            for (int j = 0;; runs++) {
                unsigned short start_pos = runs->start_pos;
                if (start_pos == 0xFFFF) {
                    runs++;
                    break;
                }
                unsigned short end_pos = runs->end_pos;
                int label = LabelsSolver::GetLabel(runs->label);

                if (label) {

                    for (unsigned short c = start_pos; c < end_pos; ++c) {

                        unsigned short needed_word_ind = c / 64;
                        if (needed_word_ind != curr_word_ind) {
                            curr_word_u = data_u[needed_word_ind];
                            curr_word_d = data_d[needed_word_ind];
                            curr_word_ind = needed_word_ind;
                        }

                        uint8_t bit_pos = c % 64;
                        uint64_t mask = 1ull << bit_pos;
                        uint64_t bit_u = curr_word_u & mask;
                        uint64_t bit_d = curr_word_d & mask;

                        if (bit_u) {
                            labels_u[c] = label;
                        }
                        if (bit_d) {
                            labels_d[c] = label;
                        }

                    }

                }
            }
        }
        if (h % 2) {
            unsigned* const labels = img_labels_.ptr<unsigned>(h - 1);
            for (int j = 0;; runs++) {
                unsigned short start_pos = runs->start_pos;
                if (start_pos == 0xFFFF) {
                    break;
                }
                unsigned short end_pos = runs->end_pos;
                int label = LabelsSolver::GetLabel(runs->label);
                for (j = start_pos; j < end_pos; j++) {
                    labels[j] = label;
                }
            }
        }
    }
};
#endif 
