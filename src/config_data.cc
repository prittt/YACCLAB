#include "config_data.h"

using namespace cv;
using namespace std;
using namespace ::filesystem;

bool ReadBool(const FileNode& node_list)
{
	bool b = false;
	if (!node_list.empty()) {
		//The entry is found
		string s((string)node_list);
		transform(s.begin(), s.end(), s.begin(), ::tolower);
		istringstream(s) >> boolalpha >> b;
	}
	return b;
}

ModeConfig::ModeConfig(string _mode, const FileNode& fn) : mode(_mode) {
	perform_blocksize		=		ReadBool(fn["perform"]["blocksize"]);
	perform_correctness		=		ReadBool(fn["perform"]["correctness"]);
	perform_average			=		ReadBool(fn["perform"]["average"]);
	perform_average_ws		=		ReadBool(fn["perform"]["average_with_steps"]);
	perform_density			=		ReadBool(fn["perform"]["density"]);
	perform_granularity		=		ReadBool(fn["perform"]["granularity"]);
	perform_memory			=		ReadBool(fn["perform"]["memory"]);

	perform_check_8connectivity_std =	ReadBool(fn["correctness_tests"]["eight_connectivity_standard"]);
	perform_check_8connectivity_ws =	ReadBool(fn["correctness_tests"]["eight_connectivity_steps"]);
	perform_check_8connectivity_mem =	ReadBool(fn["correctness_tests"]["eight_connectivity_memory"]);
	perform_check_8connectivity_bs =	ReadBool(fn["correctness_tests"]["eight_connectivity_blocksize"]);

	average_save_middle_tests =			ReadBool(fn["save_middle_tests"]["average"]);
	average_ws_save_middle_tests =		ReadBool(fn["save_middle_tests"]["average_with_steps"]);
	density_save_middle_tests =			ReadBool(fn["save_middle_tests"]["density"]);
	granularity_save_middle_tests =		ReadBool(fn["save_middle_tests"]["granularity"]);

	average_tests_number = static_cast<int>(fn["tests_number"]["average"]);
	average_ws_tests_number = static_cast<int>(fn["tests_number"]["average_with_steps"]);
	density_tests_number = static_cast<int>(fn["tests_number"]["density"]);
	granularity_tests_number = static_cast<int>(fn["tests_number"]["granularity"]);
	blocksize_tests_number = static_cast<int>(fn["tests_number"]["blocksize"]);

	density_datasets = { "classical" };
	granularity_datasets = { "granularity" };
	read(fn["check_datasets"], check_datasets);
	read(fn["average_datasets"], average_datasets);
	read(fn["average_datasets_with_steps"], average_ws_datasets);
	read(fn["memory_datasets"], memory_datasets);
	read(fn["blocksize_datasets"], blocksize_datasets);

	read(fn["algorithms"], ccl_algorithms);

	read(fn["blocksize"]["x"], user_blocksize_x);
	read(fn["blocksize"]["y"], user_blocksize_y);
	read(fn["blocksize"]["z"], user_blocksize_z);

	mode_output_path = path(mode);
}

GlobalConfig::GlobalConfig(const FileStorage& fs) {

	average_color_labels = ReadBool(fs["color_labels"]["average"]);
	density_color_labels = ReadBool(fs["color_labels"]["density"]);

	write_n_labels = ReadBool(fs["write_n_labels"]);

	input_txt = "files.txt";
	gnuplot_script_extension = ".gnuplot";
	system_script_extension =
#ifdef YACCLAB_WINDOWS
		".bat";
#elif defined(YACCLAB_LINUX) || defined(YACCLAB_UNIX) || defined(YACCLAB_APPLE)
		".sh";
#endif
	colors_folder = "colors";
	middle_folder = "middle_results";
	latex_file = "yacclab_results.tex";
	latex_charts = "averageCharts.tex";
	latex_memory_file = "memoryAccesses.tex";
	memory_file = "memory_accesses.txt";

	average_folder = "average_tests";
	average_ws_folder = "average_tests_with_steps";
	density_folder = "density_tests";
	granularity_folder = "granularity";
	memory_folder = "memory_tests";

	glob_output_path = path(fs["paths"]["output"]) / path(GetDatetimeWithoutSpecialChars());
	input_path = path(fs["paths"]["input"]);
	latex_path = path("latex");

#ifdef YACCLAB_OS
        yacclab_os = std::string(YACCLAB_OS);
#else
        yacclab_os = "";
#endif
    
    SystemInfo::set_os(yacclab_os);

}

ConfigData::ConfigData(const FileStorage& fs) : global_config(fs) {

	vector<string> modes;
	modes.push_back("CPU 2D 8-way connectivity");
	modes.push_back("CPU 3D 26-way connectivity");
    modes.push_back("CPU 2D 4-way connectivity");
	modes.push_back("CPU 3D 6-way connectivity");
#if defined YACCLAB_WITH_CUDA
	modes.push_back("GPU 2D 8-way connectivity");
	modes.push_back("GPU 3D 26-way connectivity");
    modes.push_back("GPU 2D 4-way connectivity");
	modes.push_back("GPU 3D 6-way connectivity");
#endif

	for (const string& mode : modes) {

		// If "execute" is false, LocalConfig struct for current mode isn't added to local_config_map
		if (ReadBool(fs[mode.c_str()]["execute"])) {
			mode_config_vector.emplace_back(mode, fs[mode]);
		}
	}

}