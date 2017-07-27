// Copyright(c) 2016 - 2017 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

#ifndef YACCLAB_YACCLAB_TESTS_H_
#define YACCLAB_YACCLAB_TESTS_H_
#include "file_manager.h"

#include <map>

#include <opencv2/imgproc.hpp>

#include "config_data.h"
#include "file_manager.h"
#include "labeling_algorithms.h"

class YacclabTests {
public:
    YacclabTests(const ConfigData& cfg) : cfg_(cfg) {}

    void CheckPerformLabeling();
    void CheckPerformLabelingWithSteps();
    void CheckPerformLabelingMem();

    void CheckAlgorithms(); // TODO move private

    void AverageTest();
    void AverageTestWithSteps();

    //TODO: Check correctness of memory tests also

    //Other test related functions
    void LatexGenerator();
private:
    ConfigData cfg_;
    cv::Mat1d average_results_;
    std::map<cv::String, cv::Mat1d> average_ws_results_;
    std::map<std::string, cv::Mat1d> memory_accesses_;

    bool LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const filesystem::path& files_path);
    void SaveBroadOutputResults(std::map<cv::String, cv::Mat1d>& results, const std::string& o_filename, const cv::Mat1i& labels, const std::vector<std::pair<std::string, bool>>& filenames);
    void SaveBroadOutputResults(const cv::Mat1d& results, const std::string& o_filename, const cv::Mat1i& labels, const std::vector<std::pair<std::string, bool>>& filenames);
};

#endif // !YACCLAB_YACCLAB_TESTS_H_
