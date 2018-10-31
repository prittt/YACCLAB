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
#include "tests_performer.h"


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

    OutputBox ob_setconf("Reading Configuration Parameters");
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

    ob_setconf.Cmessage("Setting Configuration Parameters DONE");
    ob_setconf.CloseBox();

    // Load configuration data from yaml
    ConfigData cfg(fs);

    // Release FileStorage
    fs.release();

    /*************************************************************************/
    /*  Configuration parameters check                                       */
    /*************************************************************************/


	for (auto &mode_cfg : cfg.mode_config_vector) {

		TestsPerformer test_perf = TestsPerformer(mode_cfg, cfg.global_config, ec);

        test_perf.InitialOperations();

		// Correctness test
		if (mode_cfg.perform_correctness) {
			if (mode_cfg.perform_check_8connectivity_std) {
				test_perf.CheckPerformLabeling();
			}

			if (mode_cfg.perform_check_8connectivity_ws) {
				test_perf.CheckPerformLabelingWithSteps();
			}

			if (mode_cfg.perform_check_8connectivity_mem) {
				test_perf.CheckPerformLabelingMem();
			}
		}

		// Average test
		if (mode_cfg.perform_average) {
			test_perf.AverageTest();
		}

		// Average with steps test
		if (mode_cfg.perform_average_ws) {
			test_perf.AverageTestWithSteps();
		}

		// Density test
		if (mode_cfg.perform_density) {
			test_perf.DensityTest();
		}

		// Granularity test
		if (mode_cfg.perform_granularity) {
			test_perf.GranularityTest();
		}
		// Memory test
		if (mode_cfg.perform_memory) {
			test_perf.MemoryTest();
		}

        // There should be better places for this
		//LabelingMapSingleton::GetLabeling(mode_cfg.ccl_existing_algorithms[0])->GetInput()->Release();

		// Latex Generator
		if (mode_cfg.perform_average || mode_cfg.perform_average_ws || mode_cfg.perform_density || mode_cfg.perform_memory || mode_cfg.perform_granularity) {
			test_perf.LatexGenerator();
		}

		// Copy log file into output folder
		dmux::cout.flush();
		filesystem::copy(path(logfile), cfg.global_config.glob_output_path / mode_cfg.mode_output_path / path(logfile), ec);
	}

    return EXIT_SUCCESS;
}