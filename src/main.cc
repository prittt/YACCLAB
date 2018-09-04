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

#if defined USE_CUDA
#include "yacclab_gpu_tests.h"
#endif

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

	// Cpu
	if (cfg.cpu_ccl_algorithms.size() > 0) {
	
		for (auto& algo_name : cfg.cpu_ccl_algorithms) {
			if (!LabelingMapSingleton::Exists(algo_name)) {
				ob_setconf.Cwarning("Unable to find the algorithm '" + algo_name + "'");
			}
			else {
				cfg.cpu_ccl_existing_algorithms.push_back(algo_name);
			}
		}

		// Check if labeling methods of the specified algorithms exist
		Labeling::img_ = Mat1b(1, 1, static_cast<uchar>(0));
		for (const auto& algo_name : cfg.cpu_ccl_existing_algorithms) {
			const auto& algorithm = LabelingMapSingleton::GetLabeling(algo_name);
			if (cfg.cpu_perform_average || cfg.cpu_perform_density || cfg.cpu_perform_granularity || (cfg.cpu_perform_correctness && cfg.cpu_perform_check_8connectivity_std)) {
				try {
					algorithm->PerformLabeling();
					cfg.cpu_ccl_average_algorithms.push_back(algo_name);
				}
				catch (const runtime_error& e) {
					ob_setconf.Cwarning(algo_name + ": " + e.what());
				}
			}
			if (cfg.cpu_perform_average_ws || (cfg.cpu_perform_correctness && cfg.cpu_perform_check_8connectivity_ws)) {
				try {
					algorithm->PerformLabelingWithSteps();
					cfg.cpu_ccl_average_ws_algorithms.push_back(algo_name);
				}
				catch (const runtime_error& e) {
					ob_setconf.Cwarning(algo_name + ": " + e.what());
				}
			}
			if (cfg.cpu_perform_memory || (cfg.cpu_perform_correctness && cfg.cpu_perform_check_8connectivity_mem)) {
				try {
					vector<unsigned long int> temp;
					algorithm->PerformLabelingMem(temp);
					cfg.cpu_ccl_mem_algorithms.push_back(algo_name);
				}
				catch (const runtime_error& e) {
					ob_setconf.Cwarning(algo_name + ": " + e.what());
				}
			}
		}

		if ((cfg.cpu_perform_average || (cfg.cpu_perform_correctness && cfg.cpu_perform_check_8connectivity_std)) && cfg.cpu_ccl_average_algorithms.size() == 0) {
			ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabeling()' method, related tests will be skipped");
			cfg.cpu_perform_average = false;
			cfg.cpu_perform_check_8connectivity_std = false;
		}

		if ((cfg.cpu_perform_average_ws || (cfg.cpu_perform_correctness && cfg.cpu_perform_check_8connectivity_ws)) && cfg.cpu_ccl_average_ws_algorithms.size() == 0) {
			ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingWithSteps()' method, related tests will be skipped");
			cfg.cpu_perform_average_ws = false;
			cfg.cpu_perform_check_8connectivity_ws = false;
		}

		if ((cfg.cpu_perform_memory || (cfg.cpu_perform_correctness && cfg.cpu_perform_check_8connectivity_mem)) && cfg.cpu_ccl_mem_algorithms.size() == 0) {
			ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingMem()' method, related tests will be skipped");
			cfg.cpu_perform_memory = false;
			cfg.cpu_perform_check_8connectivity_mem = false;
		}

		if (cfg.cpu_perform_average && (cfg.average_tests_number < 1 || cfg.average_tests_number > 999)) {
			ob_setconf.Cwarning("'average test' repetitions cannot be less than 1 or more than 999, skipped");
			cfg.cpu_perform_average = false;
		}

		if (cfg.cpu_perform_density && (cfg.density_tests_number < 1 || cfg.density_tests_number > 999)) {
			ob_setconf.Cwarning("'density test' repetitions cannot be less than 1 or more than 999, skipped");
			cfg.cpu_perform_density = false;
		}

		if (cfg.cpu_perform_average_ws && (cfg.average_ws_tests_number < 1 || cfg.average_ws_tests_number > 999)) {
			ob_setconf.Cwarning("'average with steps test' repetitions cannot be less than 1 or more than 999, skipped");
			cfg.cpu_perform_average_ws = false;
		}

		if ((cfg.cpu_perform_correctness) && cfg.check_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'correctness test', skipped");
			cfg.cpu_perform_correctness = false;
		}

		if ((cfg.cpu_perform_average) && cfg.average_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'average test', skipped");
			cfg.cpu_perform_average = false;
		}

		if ((cfg.cpu_perform_average_ws) && cfg.average_ws_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'average with steps test', skipped");
			cfg.cpu_perform_average_ws = false;
		}

		if ((cfg.cpu_perform_memory) && cfg.memory_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'memory test', skipped");
			cfg.cpu_perform_memory = false;
		}

		if (!cfg.cpu_perform_average && !cfg.cpu_perform_correctness &&
			!cfg.cpu_perform_density && !cfg.cpu_perform_memory &&
			!cfg.cpu_perform_average_ws && !cfg.cpu_perform_granularity) {
			ob_setconf.Cerror("There are no tests to perform");
		}

		// Check datasets existence
		{
			std::function<bool(vector<String>&, bool)> CheckDatasetExistence = [&cfg, &ob_setconf, &ec](vector<String>& dataset, bool print_message) -> bool {
				// Check if all the datasets' files.txt exist
				bool exists_one_dataset = false;
				for (auto& x : dataset) {
					path p = cfg.input_path / path(x) / path(cfg.input_txt);
					if (!exists(p, ec)) {
						if (print_message) {
							ob_setconf.Cwarning("There is no dataset '" + x + "' (no files.txt available), skipped");
						}
					}
					else {
						exists_one_dataset = true;
					}
				}
				return exists_one_dataset;
			};

			vector<String> ds;
			if (cfg.cpu_perform_correctness) {
				ds.insert(ds.end(), cfg.check_datasets.begin(), cfg.check_datasets.end());
			}
			if (cfg.cpu_perform_memory) {
				ds.insert(ds.end(), cfg.memory_datasets.begin(), cfg.memory_datasets.end());
			}
			if (cfg.cpu_perform_average) {
				ds.insert(ds.end(), cfg.average_datasets.begin(), cfg.average_datasets.end());
			}
			if (cfg.cpu_perform_average) {
				ds.insert(ds.end(), cfg.average_ws_datasets.begin(), cfg.average_ws_datasets.end());
			}
			std::sort(ds.begin(), ds.end());
			ds.erase(unique(ds.begin(), ds.end()), ds.end());
			CheckDatasetExistence(ds, true); // To check single dataset

			if (cfg.cpu_perform_correctness) {
				if (!CheckDatasetExistence(cfg.check_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'correctness test', skipped");
					cfg.cpu_perform_correctness = false;
				}
			}

			if (cfg.cpu_perform_average) {
				if (!CheckDatasetExistence(cfg.average_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'average test', skipped");
					cfg.cpu_perform_average = false;
				}
			}

			if (cfg.cpu_perform_average_ws) {
				if (!CheckDatasetExistence(cfg.average_ws_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'average with steps test', skipped");
					cfg.cpu_perform_average_ws = false;
				}
			}

			if (cfg.cpu_perform_memory) {
				if (!CheckDatasetExistence(cfg.memory_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'memory test', skipped");
					cfg.cpu_perform_memory = false;
				}
			}
		}

		if (cfg.cpu_perform_average || cfg.cpu_perform_average_ws || cfg.cpu_perform_density || cfg.cpu_perform_memory || cfg.cpu_perform_granularity) {
			// Set and create current output directory
			if (!create_directories(cfg.output_path, ec)) {
				ob_setconf.Cerror("Unable to create output directory '" + cfg.output_path.string() + "' - " + ec.message());
			}

			// Create the directory for latex reports
			if (!create_directories(cfg.latex_cpu_path, ec)) {
				ob_setconf.Cerror("Unable to create output directory '" + cfg.latex_cpu_path.string() + "' - " + ec.message());
			}
		}

		ob_setconf.Cmessage("Setting Configuration Parameters DONE");
		ob_setconf.CloseBox();

		YacclabTests yt(cfg);

		// Correctness test
		if (cfg.cpu_perform_correctness) {
			if (cfg.cpu_perform_check_8connectivity_std) {
				yt.CheckPerformLabeling();
			}

			if (cfg.cpu_perform_check_8connectivity_ws) {
				yt.CheckPerformLabelingWithSteps();
			}

			if (cfg.cpu_perform_check_8connectivity_mem) {
				yt.CheckPerformLabelingMem();
			}
		}

		// Average test
		if (cfg.cpu_perform_average) {
			yt.AverageTest();
		}

		// Average with steps test
		if (cfg.cpu_perform_average_ws) {
			yt.AverageTestWithSteps();
		}

		// Density test
		if (cfg.cpu_perform_density) {
			yt.DensityTest();
		}

		// Granularity test
		if (cfg.cpu_perform_granularity) {
			yt.GranularityTest();
		}
		// Memory test
		if (cfg.cpu_perform_memory) {
			yt.MemoryTest();
		}


		// Latex Generator
		if (cfg.cpu_perform_average || cfg.cpu_perform_average_ws || cfg.cpu_perform_density || cfg.cpu_perform_memory || cfg.cpu_perform_granularity) {
			yt.LatexGenerator();
		}

		// Copy log file into output folder
		dmux::cout.flush();
		filesystem::copy(path(logfile), cfg.output_path / path(logfile), ec);
	}


#if defined USE_CUDA
	// Gpu
	if (cfg.gpu_ccl_algorithms.size() > 0) {

		for (auto& algo_name : cfg.gpu_ccl_algorithms) {
			if (!LabelingMapSingleton::Exists(algo_name)) {
				ob_setconf.Cwarning("Unable to find the algorithm '" + algo_name + "'");
			}
			else {
				cfg.gpu_ccl_existing_algorithms.push_back(algo_name);
			}
		}

		// Check if labeling methods of the specified algorithms exist
		GpuLabeling::d_img_ = cuda::GpuMat(1, 1, CV_8UC1);
		for (const auto& algo_name : cfg.gpu_ccl_existing_algorithms) {
			const auto& algorithm = LabelingMapSingleton::GetLabeling(algo_name);
			if (cfg.gpu_perform_average || cfg.gpu_perform_density || cfg.gpu_perform_granularity || (cfg.gpu_perform_correctness && cfg.gpu_perform_check_8connectivity_std)) {
				try {
					algorithm->PerformLabeling();
					cfg.gpu_ccl_average_algorithms.push_back(algo_name);
				}
				catch (const runtime_error& e) {
					ob_setconf.Cwarning(algo_name + ": " + e.what());
				}
			}
			if (cfg.gpu_perform_average_ws || (cfg.gpu_perform_correctness && cfg.gpu_perform_check_8connectivity_ws)) {
				try {
					algorithm->PerformLabelingWithSteps();
					cfg.gpu_ccl_average_ws_algorithms.push_back(algo_name);
				}
				catch (const runtime_error& e) {
					ob_setconf.Cwarning(algo_name + ": " + e.what());
				}
			}
			if (cfg.gpu_perform_memory || (cfg.gpu_perform_correctness && cfg.gpu_perform_check_8connectivity_mem)) {
				try {
					vector<unsigned long int> temp;
					algorithm->PerformLabelingMem(temp);
					cfg.gpu_ccl_mem_algorithms.push_back(algo_name);
				}
				catch (const runtime_error& e) {
					ob_setconf.Cwarning(algo_name + ": " + e.what());
				}
			}
		}

		if ((cfg.gpu_perform_average || (cfg.gpu_perform_correctness && cfg.gpu_perform_check_8connectivity_std)) && cfg.gpu_ccl_average_algorithms.size() == 0) {
			ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabeling()' method, related tests will be skipped");
			cfg.gpu_perform_average = false;
			cfg.gpu_perform_check_8connectivity_std = false;
		}

		if ((cfg.gpu_perform_average_ws || (cfg.gpu_perform_correctness && cfg.gpu_perform_check_8connectivity_ws)) && cfg.gpu_ccl_average_ws_algorithms.size() == 0) {
			ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingWithSteps()' method, related tests will be skipped");
			cfg.gpu_perform_average_ws = false;
			cfg.gpu_perform_check_8connectivity_ws = false;
		}

		if ((cfg.gpu_perform_memory || (cfg.gpu_perform_correctness && cfg.gpu_perform_check_8connectivity_mem)) && cfg.gpu_ccl_mem_algorithms.size() == 0) {
			ob_setconf.Cwarning("There are no 'algorithms' with valid 'PerformLabelingMem()' method, related tests will be skipped");
			cfg.gpu_perform_memory = false;
			cfg.gpu_perform_check_8connectivity_mem = false;
		}

		if (cfg.gpu_perform_average && (cfg.average_tests_number < 1 || cfg.average_tests_number > 999)) {
			ob_setconf.Cwarning("'average test' repetitions cannot be less than 1 or more than 999, skipped");
			cfg.gpu_perform_average = false;
		}

		if (cfg.gpu_perform_density && (cfg.density_tests_number < 1 || cfg.density_tests_number > 999)) {
			ob_setconf.Cwarning("'density test' repetitions cannot be less than 1 or more than 999, skipped");
			cfg.gpu_perform_density = false;
		}

		if (cfg.gpu_perform_average_ws && (cfg.average_ws_tests_number < 1 || cfg.average_ws_tests_number > 999)) {
			ob_setconf.Cwarning("'average with steps test' repetitions cannot be less than 1 or more than 999, skipped");
			cfg.gpu_perform_average_ws = false;
		}

		if ((cfg.gpu_perform_correctness) && cfg.check_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'correctness test', skipped");
			cfg.gpu_perform_correctness = false;
		}

		if ((cfg.gpu_perform_average) && cfg.average_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'average test', skipped");
			cfg.gpu_perform_average = false;
		}

		if ((cfg.gpu_perform_average_ws) && cfg.average_ws_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'average with steps test', skipped");
			cfg.gpu_perform_average_ws = false;
		}

		if ((cfg.gpu_perform_memory) && cfg.memory_datasets.size() == 0) {
			ob_setconf.Cwarning("There are no datasets specified for 'memory test', skipped");
			cfg.gpu_perform_memory = false;
		}

		if (!cfg.gpu_perform_average && !cfg.gpu_perform_correctness &&
			!cfg.gpu_perform_density && !cfg.gpu_perform_memory &&
			!cfg.gpu_perform_average_ws && !cfg.gpu_perform_granularity) {
			ob_setconf.Cerror("There are no tests to perform");
		}

		// Check datasets existence
		{
			std::function<bool(vector<String>&, bool)> CheckDatasetExistence = [&cfg, &ob_setconf, &ec](vector<String>& dataset, bool print_message) -> bool {
				// Check if all the datasets' files.txt exist
				bool exists_one_dataset = false;
				for (auto& x : dataset) {
					path p = cfg.input_path / path(x) / path(cfg.input_txt);
					if (!exists(p, ec)) {
						if (print_message) {
							ob_setconf.Cwarning("There is no dataset '" + x + "' (no files.txt available), skipped");
						}
					}
					else {
						exists_one_dataset = true;
					}
				}
				return exists_one_dataset;
			};

			vector<String> ds;
			if (cfg.gpu_perform_correctness) {
				ds.insert(ds.end(), cfg.check_datasets.begin(), cfg.check_datasets.end());
			}
			if (cfg.gpu_perform_memory) {
				ds.insert(ds.end(), cfg.memory_datasets.begin(), cfg.memory_datasets.end());
			}
			if (cfg.gpu_perform_average) {
				ds.insert(ds.end(), cfg.average_datasets.begin(), cfg.average_datasets.end());
			}
			if (cfg.gpu_perform_average) {
				ds.insert(ds.end(), cfg.average_ws_datasets.begin(), cfg.average_ws_datasets.end());
			}
			std::sort(ds.begin(), ds.end());
			ds.erase(unique(ds.begin(), ds.end()), ds.end());
			CheckDatasetExistence(ds, true); // To check single dataset

			if (cfg.gpu_perform_correctness) {
				if (!CheckDatasetExistence(cfg.check_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'correctness test', skipped");
					cfg.gpu_perform_correctness = false;
				}
			}

			if (cfg.gpu_perform_average) {
				if (!CheckDatasetExistence(cfg.average_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'average test', skipped");
					cfg.gpu_perform_average = false;
				}
			}

			if (cfg.gpu_perform_average_ws) {
				if (!CheckDatasetExistence(cfg.average_ws_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'average with steps test', skipped");
					cfg.gpu_perform_average_ws = false;
				}
			}

			if (cfg.gpu_perform_memory) {
				if (!CheckDatasetExistence(cfg.memory_datasets, false)) {
					ob_setconf.Cwarning("There are no valid datasets for 'memory test', skipped");
					cfg.gpu_perform_memory = false;
				}
			}
		}

		if (cfg.gpu_perform_average || cfg.gpu_perform_average_ws || cfg.gpu_perform_density || cfg.gpu_perform_memory || cfg.gpu_perform_granularity) {
			// Set and create current output directory
			if (!create_directories(cfg.output_path, ec)) {
				ob_setconf.Cerror("Unable to create output directory '" + cfg.output_path.string() + "' - " + ec.message());
			}

			// Create the directory for latex reports
			if (!create_directories(cfg.latex_gpu_path, ec)) {
				ob_setconf.Cerror("Unable to create output directory '" + cfg.latex_gpu_path.string() + "' - " + ec.message());
			}
		}

		ob_setconf.Cmessage("Setting Configuration Parameters DONE");
		ob_setconf.CloseBox();

		YacclabGpuTests yt(cfg);

		// Correctness test
		if (cfg.gpu_perform_correctness) {
			if (cfg.gpu_perform_check_8connectivity_std) {
				yt.CheckPerformLabeling();
			}

			if (cfg.gpu_perform_check_8connectivity_ws) {
				yt.CheckPerformLabelingWithSteps();
			}

			if (cfg.gpu_perform_check_8connectivity_mem) {
				yt.CheckPerformLabelingMem();
			}
		}

		// Average test
		if (cfg.gpu_perform_average) {
			yt.AverageTest();
		}

		// Average with steps test
		if (cfg.gpu_perform_average_ws) {
			yt.AverageTestWithSteps();
		}

		// Density test
		if (cfg.gpu_perform_density) {
			yt.DensityTest();
		}

		// Granularity test
		if (cfg.gpu_perform_granularity) {
			yt.GranularityTest();
		}
		// Memory test
		if (cfg.gpu_perform_memory) {
			yt.MemoryTest();
		}

		GpuLabeling::d_img_.release();

		// Latex Generator
		if (cfg.gpu_perform_average || cfg.gpu_perform_average_ws || cfg.gpu_perform_density || cfg.gpu_perform_memory || cfg.gpu_perform_granularity) {
			yt.LatexGenerator();
		}

		// Copy log file into output folder
		dmux::cout.flush();
		filesystem::copy(path(logfile), cfg.output_path / path(logfile), ec);

	}
#endif

	if (cfg.cpu_ccl_algorithms.size() == 0 
#if defined USE_CUDA
		&& cfg.gpu_ccl_algorithms.size() == 0
#endif
		) {
		ob_setconf.Cerror("Empty algorithms list");
	}
	

    return EXIT_SUCCESS;
}