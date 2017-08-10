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

#include <cstdint>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "config_data.h"
#include "file_manager.h"
#include "labeling_algorithms.h"
#include "latex_generator.h"
#include "memory_tester.h"
#include "performance_evaluator.h"
#include "progress_bar.h"
#include "stream_demultiplexer.h"
#include "system_info.h"
#include "utilities.h"
#include "yacclab_tests.h"

using namespace std;
using namespace cv;

int main()
{
    // Redirect cv exceptions
    cvRedirectError(RedirectCvError);

    // Hide cursor from console
    HideConsoleCursor();

    // To handle filesystem errors
    error_code ec;

    // Create StreamDemultiplexer object in order
    // to print output on both stdout and log file
    string logfile = "log.txt";
    ofstream os(logfile);
    if (os.is_open()) {
        dmux::cout.AddStream(os);
    }

    OutputBox ob_setconf("Setting Configuration Parameters");
    // Read yaml configuration file
    const string config_file = "config.yaml";
    FileStorage fs;
    try {
        fs.open(config_file, FileStorage::READ);
    }
    catch (const cv::Exception&) {
        exit(EXIT_FAILURE);  // Error redirected,
                             // OpenCV redirected function will
                             // print the error on stdout
    }

    if (!fs.isOpened()) {
        ob_setconf.Cerror("Failed to open '" + config_file + "'");
        // EXIT_FAILURE
    }

    // Load configuration data from yaml
    ConfigData cfg(fs);

    // Release FileStorage
    fs.release();

    /*************************************************************************/
    /*  Configuration parameters check                                       */
    /*************************************************************************/

    // Check if all the specified algorithms exist
    for (auto& algo_name : cfg.ccl_algorithms) {
        if (!LabelingMapSingleton::Exists(algo_name)) {
            ob_setconf.Cwarning("Unable to find the algorithm '" + algo_name + "'");
        }
        else {
            cfg.ccl_existing_algorithms.push_back(algo_name);
        }
    }

    if (cfg.ccl_existing_algorithms.size() == 0) {
        ob_setconf.Cerror("There are no valid values in the 'algorithms' list");
    }

    // Check if labeling methods of the specified algorithms exist
    Labeling::img_ = Mat1b(1, 1, static_cast<uchar>(0));
    for (const auto& algo_name : cfg.ccl_existing_algorithms) {
        const auto& algorithm = LabelingMapSingleton::GetLabeling(algo_name);
        if (cfg.perform_average || cfg.perform_density || (cfg.perform_correctness && cfg.perform_check_8connectivity_std)) {
            try {
                algorithm->PerformLabeling();
                cfg.ccl_average_algorithms.push_back(algo_name);
            }
            catch (const runtime_error& e) {
                ob_setconf.Cwarning(algo_name + ": " + e.what());
            }
        }
        if (cfg.perform_average_ws || (cfg.perform_correctness && cfg.perform_check_8connectivity_ws)) {
            try {
                algorithm->PerformLabelingWithSteps();
                cfg.ccl_average_ws_algorithms.push_back(algo_name);
            }
            catch (const runtime_error& e) {
                ob_setconf.Cwarning(algo_name + ": " + e.what());
            }
        }
        if (cfg.perform_memory || (cfg.perform_correctness && cfg.perform_check_8connectivity_mem)) {
            try {
                algorithm->PerformLabelingMem(vector<unsigned long int>{});
                cfg.ccl_mem_algorithms.push_back(algo_name);
            }
            catch (const runtime_error& e) {
                ob_setconf.Cwarning(algo_name + ": " + e.what());
            }
        }
    }

    if ((cfg.perform_average || (cfg.perform_correctness && cfg.perform_check_8connectivity_std)) && cfg.ccl_average_algorithms.size() == 0) {
        ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabeling()' method, related tests will be skipped");
        cfg.perform_average = false;
        cfg.perform_check_8connectivity_std = false;
    }

    if ((cfg.perform_average_ws || (cfg.perform_correctness && cfg.perform_check_8connectivity_ws)) && cfg.ccl_average_ws_algorithms.size() == 0) {
        ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingWithSteps()' method, related tests will be skipped");
        cfg.perform_average_ws = false;
        cfg.perform_check_8connectivity_ws = false;
    }

    if ((cfg.perform_memory || (cfg.perform_correctness && cfg.perform_check_8connectivity_mem)) && cfg.ccl_mem_algorithms.size() == 0) {
        ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingMem()' method, related tests will be skipped");
        cfg.perform_memory = false;
        cfg.perform_check_8connectivity_mem = false;
    }

    if (cfg.perform_average && (cfg.average_tests_number < 1 || cfg.average_tests_number > 999)) {
        ob_setconf.Cwarning("'Average tests' repetitions cannot be less than 1 or more than 999, skipped");
        cfg.perform_average = false;
    }

    if (cfg.perform_density && (cfg.density_tests_number < 1 || cfg.density_tests_number > 999)) {
        ob_setconf.Cwarning("'Density tests' repetitions cannot be less than 1 or more than 999, skipped");
        cfg.perform_density = false;
    }

    if (cfg.perform_average_ws && (cfg.average_ws_tests_number < 1 || cfg.average_ws_tests_number > 999)) {
        ob_setconf.Cwarning("'Average tests with steps' repetitions cannot be less than 1 or more than 999, skipped");
        cfg.perform_average_ws = false;
    }

    if ((cfg.perform_correctness) && cfg.check_datasets.size() == 0) {
        ob_setconf.Cwarning("There are no datasets specified for 'correctness tests', skipped");
        cfg.perform_correctness = false;
    }

    if ((cfg.perform_average) && cfg.average_datasets.size() == 0) {
        ob_setconf.Cwarning("There are no datasets specified for 'average tests', skipped");
        cfg.perform_average = false;
    }

    if ((cfg.perform_memory) && cfg.memory_datasets.size() == 0) {
        ob_setconf.Cwarning("There are no datasets specified for 'memory tests', skipped");
        cfg.perform_memory = false;
    }

    if (!cfg.perform_average && !cfg.perform_correctness &&
        !cfg.perform_density && !cfg.perform_memory &&
        !cfg.perform_average_ws) {
        ob_setconf.Cerror("There are no tests to perform");
    }

    // Check datasets
    vector<String> ds(cfg.memory_datasets);
    ds.insert(ds.end(), cfg.average_datasets.begin(), cfg.average_datasets.end());
    ds.insert(ds.end(), cfg.check_datasets.begin(), cfg.check_datasets.end());
    std::sort(ds.begin(), ds.end());
    ds.erase(unique(ds.begin(), ds.end()), ds.end());

    // Check if all the datasets files.txt exist
    for (auto& x : ds) {
        path p = cfg.input_path / path(x) / path(cfg.input_txt);
        if (!exists(p, ec)) {
            ob_setconf.Cwarning("There is no dataset (no files.txt available) " + p.string() + ", skipped");
        }
    }

    // Set and create current output directory
    if (!create_directories(cfg.output_path, ec)) {
        ob_setconf.Cerror("Unable to create output directory '" + cfg.output_path.string() + "' - " + ec.message());
    }

    // Create the directory for latex reports
    if (!create_directories(cfg.latex_path, ec)) {
        ob_setconf.Cerror("Unable to create output directory '" + cfg.latex_path.string() + "' - " + ec.message());
    }

    ob_setconf.Cmessage("Setting Configuration Parameters DONE");
    ob_setconf.CloseBox();

    YacclabTests yt(cfg);

    // Correctness test
    if (cfg.perform_correctness) {
        if (cfg.perform_check_8connectivity_std) {
            yt.CheckPerformLabeling();
        }

        if (cfg.perform_check_8connectivity_ws) {
            yt.CheckPerformLabelingWithSteps();
        }

        if (cfg.perform_check_8connectivity_mem) {
            yt.CheckPerformLabelingMem();
        }
    }

    // Average tests
    if (cfg.perform_average) {
        yt.AverageTest();
    }

    // Average with steps tests
    if (cfg.perform_average_ws) {
        yt.AverageTestWithSteps();
    }

    // Density tests
    if (cfg.perform_density) {
        yt.DensityTest();
    }

    // Granularity tests

    // Memory tests
    if (cfg.perform_memory) {
        yt.MemoryTest();
    }

    // Latex Generator
    yt.LatexGenerator();

    // Copy log file into output folder
    dmux::cout.flush();
    copy(path(logfile), cfg.output_path / path(logfile), ec);

    return EXIT_SUCCESS;
}