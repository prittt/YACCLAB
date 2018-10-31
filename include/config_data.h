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

#ifndef YACCLAB_CONFIG_DATA_H_
#define YACCLAB_CONFIG_DATA_H_

#include <opencv2/imgproc.hpp>
#include <string>
#include <map>

#include "file_manager.h"
#include "system_info.h"
#include "utilities.h"

using namespace filesystem;

bool ReadBool(const cv::FileNode& node_list);

struct ModeConfig {

	std::string mode;

	bool perform_correctness;            // Whether to perform correctness tests or not
	bool perform_average;                // Whether to perform average tests or not
	bool perform_density;                // Whether to perform density tests or not
	bool perform_granularity;            // Whether to perform granularity tests or not
	bool perform_memory;                 // Whether to perform memory tests or not
	bool perform_average_ws;             // Whether to perform average tests with steps or not

	bool perform_check_8connectivity_std;	// Whether to perform 8-connectivity test on PerformLabeling() functions
	bool perform_check_8connectivity_ws;	// Whether to perform 8-connectivity test on PerformLabelingWithSteps() functions
	bool perform_check_8connectivity_mem;	// Whether to perform 8-connectivity test on PerformLabelingMem() functions

	bool average_save_middle_tests;      // If true, results of each average test run will be stored 
	bool density_save_middle_tests;      // If true, results of each density test run will be stored 
	bool granularity_save_middle_tests;  // If true, results of each granularity test run will be stored 
	bool average_ws_save_middle_tests;   // If true, results of each average test with steps run will be stored 

	unsigned average_tests_number;        // Reps of average tests (only the minimum will be considered)
	unsigned average_ws_tests_number;     // Reps of average tests with steps (only the minimum will be considered)
	unsigned density_tests_number;        // Reps of density tests (only the minimum will be considered)
	unsigned granularity_tests_number;    // Reps of density tests (only the minimum will be considered)

	std::vector<std::string> check_datasets;       // List of datasets on which check tests will be performed
	std::vector<std::string> memory_datasets;      // List of datasets on which memory tests will be perform
	std::vector<std::string> density_datasets;     // List of datasets on which density tests will be performed
	std::vector<std::string> granularity_datasets; // List of datasets on which granularity tests will be performed
	std::vector<std::string> average_datasets;     // Lists of dataset on which average tests will be performed
	std::vector<std::string> average_ws_datasets;  // Lists of dataset on which average tests whit steps will be performed

	std::vector<std::string> ccl_algorithms;          // Lists of algorithms specified by the user in the config.yaml
	std::vector<std::string> ccl_existing_algorithms; // Lists of 'ccl_algorithms' actually existing

	std::vector<std::string> ccl_mem_algorithms;        // List of algorithms which actually support memory tests
	std::vector<std::string> ccl_average_algorithms;    // List of algorithms which actually support average tests
	std::vector<std::string> ccl_average_ws_algorithms; // List of algorithms which actually support average with steps tests

	path mode_output_path;

	ModeConfig(std::string _mode, const cv::FileNode& fn);
};

struct GlobalConfig {

	std::string input_txt;						// File of images list
	std::string gnuplot_script_extension;		// Gnuplot scripts extension
	std::string system_script_extension;		// System-based scripts extension
	std::string colors_folder;					// Folder which will store colored images
	std::string middle_folder;					// Folder which will store middle results
	std::string latex_file;						// Latex file which will store textual average results
	std::string latex_memory_file;				// Latex file which will store textual memory results
	std::string latex_charts;					// Latex file which will store report latex code for charts
	std::string memory_file;					// File which will store report textual memory results

	std::string average_folder;					// Folder which will store average test results
	std::string average_ws_folder;				// Folder which will store average test with steps results
	std::string density_folder;					// Folder which will store density results
	std::string granularity_folder;				// Folder which will store granularity results
	std::string memory_folder;					// Folder which will store memory results

	path glob_output_path;						// Path on which results are stored
	path input_path;							// Path on which input datasets are stored

	std::string yacclab_os;						// Name of the current OS

	// Verranno eliminati o almeno cambiati
	bool average_color_labels;           // If true, labeled image from average tests will be colored and stored
	bool density_color_labels;           // If true, labeled image from density tests will be colored and stored
	bool write_n_labels;                 // If true, the number of components generated by the algorithms will be stored in the output file

	// da decidere dove e in che forma infilare   
	path latex_path;                      // Path on which latex report will be stored

	GlobalConfig(const cv::FileStorage& fs);
};

struct ConfigData {

	std::vector<ModeConfig> mode_config_vector;

	GlobalConfig global_config;

	ConfigData(const cv::FileStorage& fs);

};


#endif // !YACCLAB_CONFIG_DATA_H_
