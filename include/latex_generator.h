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

#ifndef YACCLAB_LATEX_GENERATOR_H_
#define YACCLAB_LATEX_GENERATOR_H_

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "file_manager.h"

void GenerateLatexTable(const filesystem::path& output_path, const std::string& latex_file, const cv::Mat1d& all_res,
    const std::vector<cv::String>& alg_name, const std::vector<cv::String>& ccl_algorithms);

void GenerateMemoryLatexTable(const filesystem::path& output_path, const std::string& latex_file, const cv::Mat1d& accesses,
    const std::string& dataset, const std::vector<cv::String>& ccl_mem_algorithms);

void GenerateLatexCharts(const filesystem::path& output_path, const std::string& latex_charts, const std::string& latex_folder,
    const std::vector<cv::String>& datasets_name);

void LatexGenerator(const std::map<std::string, bool>& test_to_perform, const filesystem::path& output_path, const std::string& latex_file,
    const cv::Mat1d& all_res, const std::vector<cv::String>& datasets_name, const std::vector<cv::String>& ccl_algorithms,
    const std::vector<cv::String>& ccl_mem_algorithms, const std::map<std::string, cv::Mat1d>& accesses);

#endif // !YACCLAB_LATEX_GENERATOR_H_