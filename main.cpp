#include <opencv2\core\core.hpp> 
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <deque>          
#include <list>           
#include <queue>
#include <string>
#include <array> 
#include <algorithm>
#include <functional>

using namespace cv;
using namespace std;

#include "performanceEvaluator.h"
#include "configurationReader.h"
#include "labelingAlgorithms.h"

#include <sys/types.h>
#include <sys/stat.h>

bool dirExists(const char* pathname){
	struct stat info;
	if (stat(pathname, &info) != 0){
		//printf("cannot access %s\n", pathname);
		return false; 
	}
	else if (info.st_mode & S_IFDIR){  // S_ISDIR() doesn't exist on my windows 
		//printf("%s is a directory\n", pathname);
		return true;
	}
	
	//printf("%s is no directory\n", pathname);
	return false; 
}

// TODO if necessary
void syntax() {
	cerr << "" << endl;
	exit(EXIT_FAILURE);
}
void error(const string& s) {
	cerr << s;
	exit(EXIT_FAILURE);
}
// TODO if necessary

// Create a bunch of pseudo random colors from labels indexes and create a
// color representation for the labels
void colorLabels(const Mat1i& imgLabels, Mat3b& imgOut) {
	imgOut = Mat3b(imgLabels.size());
	for (int r = 0; r<imgLabels.rows; ++r) {
		for (int c = 0; c<imgLabels.cols; ++c) {
			imgOut(r, c) = Vec3b(imgLabels(r, c) * 131 % 255, imgLabels(r, c) * 241 % 255, imgLabels(r, c) * 251 % 255);
		}
	}
}

// This function may be useful to compare the output of different labeling procedures
// which may assign different labels to the same object. Use this to force a row major
// ordering of labels.
void normalizeLabels(Mat1i& imgLabels, int iNumLabels) {
	vector<int> vecNewLabels(iNumLabels + 1, 0);
	int iMaxNewLabel = 0;

	for (int r = 0; r<imgLabels.rows; ++r) {
		for (int c = 0; c<imgLabels.cols; ++c) {
			int iCurLabel = imgLabels(r, c);
			if (iCurLabel>0) {
				if (vecNewLabels[iCurLabel] == 0) {
					vecNewLabels[iCurLabel] = ++iMaxNewLabel;
				}
				imgLabels(r, c) = vecNewLabels[iCurLabel];
			}
		}
	}
}

// Get binary image given a image's FileName; 
bool getBinaryImage(const string FileName, Mat1b& binaryMat){

	// Image load
	Mat image;
    image = imread(FileName, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    // Check if image exist
	if (image.empty())
		return false;

	//// Convert the image to grayscale
	//Mat grayscaleMat;
	//cvtColor(image, grayscaleMat, CV_RGB2GRAY);

	// Adjust the threshold to actually make it binary
	threshold(image, binaryMat, 100, 1, CV_THRESH_BINARY);

	return true;
}

bool compareMat(const Mat1i& mata, const Mat1i& matb){

    // Get a matrix with non-zero values at points where the 
    // two matrices have different values
    cv::Mat diff = mata != matb;
    // Equal if no elements disagree
    return cv::countNonZero(diff) == 0;
}

void checkAlgorithms(vector<pair<CCLPointer, string>>& CCLAlgorithms, const vector<string>& datasets, const string& input_path, const string& input_txt){

    vector<bool> stats(CCLAlgorithms.size(), true); // true if the i-th algorithm is correct, false otherwise
    bool stop = false; // true if all algorithms are incorrect

    for (uint i = 0; i < datasets.size(); ++i){
        // For every dataset in check list

        string is_path = input_path + "\\" + datasets[i] + "\\" + input_txt;
        ifstream is(is_path);
        if (!is.is_open()){
            cout << "Unable to open " + is_path << endl;
            continue;
        }

        while (true && !stop){
            string filename;
            getline(is, filename);
            if (is.eof())
                break;

            Mat1b binaryImg;
            Mat1i labeledImgCorrect, labeledImgToControl;
            unsigned nLabelsCorrect, nLabelsToControl;

            if (!getBinaryImage(input_path + "\\" + datasets[i] + "\\" + filename, binaryImg)){
                cout << "Unable to check on " + filename + ", file does not exist" << endl;
                continue;
            }

            nLabelsCorrect = connectedComponents(binaryImg, labeledImgCorrect); // OPENCV is the reference
            uint j = 0; 
            for (vector<pair<CCLPointer, string>>::iterator it = CCLAlgorithms.begin(); it != CCLAlgorithms.end(); ++it, ++j){
                // For all the Algorithms in the array
                
                if (stats[j]){
                    nLabelsToControl = (*it).first(binaryImg, labeledImgToControl);
                    normalizeLabels(labeledImgToControl, nLabelsToControl);
                    if (nLabelsCorrect != nLabelsToControl || !compareMat(labeledImgCorrect, labeledImgToControl)){
                        stats[j] = false;
                        cout << "\"" << (*it).second << "\" is not correct, it fails on " << input_path + "\\" + datasets[i] + "\\" + filename << endl;
                        if (adjacent_find(stats.begin(), stats.end(), not_equal_to<int>()) == stats.end()){
                            stop = true; 
                            break;
                        }
                    }
                }
            }
        }
    }

    uint j = 0;
    for (vector<pair<CCLPointer, string>>::iterator it = CCLAlgorithms.begin(); it != CCLAlgorithms.end(); ++it, ++j){
        if (stats[j]){
            cout << "\"" << (*it).second << "\" is correct!" << endl;
        }
    }
}

int labelingOPENCV(const Mat1b& binaryMat, Mat1i& labeledMat){
	return connectedComponents(binaryMat, labeledMat, 8, CV_32S);
}


// This function take a char as input and return the corresponding int value (not ASCII one)
unsigned int ctoi(const char &c){
	return ((int)c - 48);
}

string averages_test(vector<pair<CCLPointer, string>>& CCLAlgorithms, const string& input_path, const string& input_folder, const string& input_txt, const string& gnuplot_script, string& output_path, string& colors_folder, const bool& write_n_labels = true, const bool& output_colors = true){

	string output_folder = input_folder,
		   complete_output_path = output_path + "\\" + output_folder,
		   output_broad_results = "results.txt",
		   output_averages_results = "averages.txt",
		   output_graph = output_folder + ".pdf",
           output_graph_bw = output_folder + "_bw.pdf";

	if (!dirExists(complete_output_path.c_str()))
		if (0 != std::system(("mkdir " + complete_output_path).c_str()))
			return ("Averages_Test on '" + input_folder + "': Unable to find/create the output path " + complete_output_path);

	string is_path = input_path + "\\" + input_folder + "\\" + input_txt,
		   os_path = output_path + "\\" + output_folder + "\\" + output_broad_results,
		   averages_os_path = output_path + "\\" + output_folder + "\\" + output_averages_results;

	// For LIST OF INPUT IMAGES
	ifstream is(is_path);
	if (!is.is_open())
		return ("Averages_Test on '" + input_folder + "': Unable to open " + is_path);
	// For BROAD RESULT
	ofstream os(os_path);
	if (!os.is_open())
		return ("Averages_Test on '" + input_folder + "': Unable to open " + os_path);
	// For AVERAGES RESULT
	ofstream averages_os(averages_os_path);
	if (!averages_os.is_open())
		return ("Averages_Test on '" + input_folder + "': Unable to open " + averages_os_path);

	// To set heading file format (BROAD + AVERAGES)
	//averages_os << "#";
	os << "#";
	for (vector<pair<CCLPointer, string>>::iterator it = CCLAlgorithms.begin(); it != CCLAlgorithms.end(); ++it){
		os << "\t" << (*it).second;
		write_n_labels ? os << "\t" << "n_label" : os << "";
		//averages_os << "\t" << (*it).second;
	}
	os << endl;
	//averages_os << endl;
	// To set heading file format (BROAD + AVERAGES)

	vector<pair<double, uint16_t>> supp_averages(CCLAlgorithms.size(),  make_pair(0, 0));

	PerformanceEvaluator perf;
	while (true){
		string filename;
		getline(is, filename);
		if (is.eof())
			break;

		Mat1b binaryImg;

		if (!getBinaryImage(input_path + "\\" + input_folder + "\\" + filename, binaryImg)){
			cout << filename + " does not exist" << endl;
			continue;
		}

		os << filename;

		unsigned int i = 0; 
		for (vector<pair<CCLPointer, string>>::iterator it = CCLAlgorithms.begin(); it != CCLAlgorithms.end(); ++it, ++i){
			// For all the Algorithms in the array

			// This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
			Mat1i labeledMat;
			unsigned nLabels;
			Mat3b imgColors;

			perf.start((*it).second);
			nLabels = (*it).first(binaryImg, labeledMat);
			perf.stop((*it).second);

			os << "\t" << perf.last((*it).second);
			write_n_labels ? os << "\t" << nLabels : os << "";

			if (output_colors){
                string out_folder = output_path + "\\" + output_folder + "\\" + colors_folder;
                if (!dirExists(out_folder.c_str()))
                    if (0 != std::system(("mkdir " + out_folder).c_str()))
                        return ("Averages_Test on '" + input_folder + "': Unable to find/create the output path " + out_folder);
                
                normalizeLabels(labeledMat, nLabels);
				colorLabels(labeledMat, imgColors);
                imwrite(out_folder + "\\" + filename + "_" + (*it).second + ".png", imgColors);
			}

			// 
			supp_averages[i].first += perf.last((*it).second);
			supp_averages[i].second++;

		}
		os << endl;
	}

	// To calculate averages times and write it on the specified file
	// averages_os << "average_time"; 
	for (unsigned int i = 0; i < CCLAlgorithms.size(); ++i){
		// For all the Algorithms in the array
		averages_os << CCLAlgorithms[i].second << "\t" << supp_averages[i].first/supp_averages[i].second << endl;
	} 

	// GNUPLOT SCRIPT
	string scriptos_path = output_path + "\\" + output_folder + "\\" + gnuplot_script;
	ofstream scriptos(scriptos_path);
	if (!scriptos.is_open())
		return ("Averages_Test on '" + input_folder + "': Unable to create " + scriptos_path);

    scriptos << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl; 
    scriptos << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << gnuplot_script << "' if you want to run it" << endl << endl;
    
    scriptos << "reset" << endl;   
	scriptos << "cd '" << complete_output_path << "\'" << endl;
    scriptos << "set grid ytic" << endl; 
    scriptos << "set grid" << endl << endl; 

    scriptos << "# " << output_folder << "(COLORS)" << endl;
    scriptos << "set output \"" + output_graph + "\"" << endl;
    scriptos << "set title \"" + output_folder + "\" font ', 12'" << endl << endl;
    
    scriptos << "# pdf colors" << endl;
    scriptos << "set terminal pdf enhanced color font ',10'" << endl << endl;

    scriptos << "# Graph style"<< endl;
    scriptos << "set style data histogram" << endl;
	scriptos << "set style histogram cluster gap 1" << endl;
	scriptos << "set style fill solid border -1" << endl;
	scriptos << "set boxwidth 0.9" << endl << endl;
	
    scriptos << "# Get stats to set labels" << endl;
    scriptos << "stats \"" << output_averages_results << "\" using 2 nooutput" << endl;
    scriptos << "ymax = STATS_max + (STATS_max/100)*10" << endl;
    scriptos << "xw = 0" << endl;
    scriptos << "yw = (ymax)/22" << endl << endl;

    scriptos << "# Axes labels" << endl;
    scriptos << "set xtic rotate by -45 scale 0" << endl;
    scriptos << "set ylabel \"Execution Time [ms]\"" << endl << endl;

    scriptos << "# Axes range" << endl; 
    scriptos << "set yrange[0:ymax]" << endl;
    scriptos << "set xrange[*:*]" << endl << endl;

    scriptos << "# Legend" << endl; 
    scriptos << "set key off" << endl << endl;

    scriptos << "# Plot" << endl; 
	scriptos << "plot \\" << endl; 
    scriptos << "\"" + output_averages_results + "\" using 2:xtic(1), \"" << output_averages_results << "\" using ($0 - xw) : ($2 + yw) : (stringcolumn(2)) with labels" << endl << endl;
	
    scriptos << "# " << output_folder << "(BLACK AND WHITE)" << endl;
    scriptos << "set output \"" + output_graph_bw + "\"" << endl;
    scriptos << "set title \"" + output_folder + "\" font ', 12'" << endl << endl;

    scriptos << "# pdf black and white" << endl;
    scriptos << "set terminal pdf enhanced monochrome dashed font ',10'" << endl << endl;

    scriptos << "replot" << endl << endl;

    scriptos << "exit gnuplot" << endl;
	
	averages_os.close();
	scriptos.close();
    // GNUPLOT SCRIPT

    if( 0 != std::system(("gnuplot " + complete_output_path + "\\" + gnuplot_script).c_str()))
        return ("Averages_Test on '" + input_folder + "': Unable to run gnuplot's script");
	return ("Averages_Test on '" + input_folder + "': successfuly done");
}

string density_size_test(vector<pair<CCLPointer,string>>& CCLAlgorithms, const string& input_path, const string& input_folder, const string& input_txt, const string& gnuplot_script, string& output_path, string& colors_folder, const bool& write_n_labels = true, const bool& output_colors = true){
	
	string output_folder = input_folder,
		   complete_output_path = output_path + "\\" + output_folder,
		   output_broad_result = "results.txt",
		   output_size_result = "size.txt",
		   output_density_result = "density.txt",
		   output_size_graph = "size.pdf",
           output_size_graph_bw = "size_bw.pdf",
		   output_density_graph = "density.pdf",
           output_density_graph_bw = "density_bw.pdf";	

	if (!dirExists(complete_output_path.c_str()))
		if (0 != std::system(("mkdir " + complete_output_path).c_str()))
			return ("Density_Size_Test on '" + input_folder + "': Unable to find/create the output path " + complete_output_path); 

	string is_path = input_path + "\\" + input_folder + "\\" + input_txt,
		   os_path = output_path + "\\" + output_folder + "\\" + output_broad_result,
		   density_os_path = output_path + "\\" + output_folder + "\\" + output_density_result,
	       size_os_path = output_path + "\\" + output_folder + "\\" + output_size_result;

	// For LIST OF INPUT IMAGES
	ifstream is(is_path);
	if(!is.is_open())
		return ("Density_Size_Test on '" + input_folder + "': Unable to open " + is_path);
	// For BROAD RESULT
	ofstream os(os_path);
	if (!os.is_open())
		return ("Density_Size_Test on '" + input_folder + "': Unable to create " + os_path);
	// For DENSITY RESULT
	ofstream density_os(density_os_path);
	if (!density_os.is_open())
		return ("Density_Size_Test on '" + input_folder + "': Unable to create " + density_os_path);
	// For SIZE RESULT
	ofstream size_os(size_os_path);
	if (!size_os.is_open())
		return ("Density_Size_Test on '" + input_folder + "': Unable to create " + size_os_path);

	// To set heading file format (BROAD RESULT, SIZE RESULT, DENSITY RESULT)
	os << "#";
	density_os << "#Density";
	size_os << "#Size";
	for (vector<pair<CCLPointer, string>>::iterator it = CCLAlgorithms.begin(); it != CCLAlgorithms.end(); ++it){
		os << "\t" << (*it).second;
		write_n_labels ? os << "\t" << "n_label" : os << "";
		density_os << "\t" << (*it).second;
		size_os << "\t" << (*it).second;
	}
	os << endl;
	density_os << endl;
	size_os << endl;
	// To set heading file format (BROAD RESULT, SIZE RESULT, DENSITY RESULT)

	PerformanceEvaluator perf;

	uint8_t density = 9 /*[0.1,0.9]*/, size = 8 /*[32,64,128,256,512,1024,2048,4096]*/;

	vector<vector<pair<double, uint16_t>>> supp_density(CCLAlgorithms.size(), vector<pair<double, uint16_t>>(density, make_pair(0, 0)));
	vector<vector<pair<double, uint16_t>>> supp_size(CCLAlgorithms.size(), vector<pair<double, uint16_t>>(size, make_pair(0, 0)));
	// Note that number of random_images is less than 800, this is why the second element of the 
	// pair has uint16_t data type. Extern vector represent the algorithms, inner vector represent 
	// density for "supp_density" variable and dimension for "supp_dimension" one. In particular: 
	//	
	//	FOR "supp_density" VARIABLE:	
	//	INNER_VECTOR[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
	//	INNER_VECTPR[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
	//  .. and so on;
	//
	//	SO: 
	//	  supp_density[0][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
	//	  for algorithm in position 0;
	//	  
	//	  supp_density[0][1] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
	//	  for algorithm in position 0;
	//	
	//	  supp_density[1][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
	//	  for algorithm in position 1;
	//    .. and so on
	//
	//	FOR "SUP_DIMENSION VARIABLE": 
	//	INNER_VECTOR[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_32*32_DIMENSION, COUNT_OF_THAT_IMAGES }
	//	INNER_VECTOR[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_64*64_DIMENSION, COUNT_OF_THAT_IMAGES }
	//
	//	view "supp_density" explanation for more details;
	//

	while (true){
		string filename;
		getline(is, filename);
		if (is.eof())
			break;

		Mat1b binaryImg;

		if (!getBinaryImage(input_path + "\\" + input_folder + "\\" + filename, binaryImg)){
			cout << filename + " does not exist" << endl;
			continue;
		}

		os << filename;

		unsigned int i = 0;
		for (vector<pair<CCLPointer, string>>::iterator it = CCLAlgorithms.begin(); it != CCLAlgorithms.end(); ++it, ++i){
			// For all the Algorithms in the array
			
			// This variable need to be redefined for every algorithms to uniform performance result (in particular this is true for labeledMat?)
			Mat1i labeledMat;
			unsigned nLabels;
			Mat3b imgColors;
			
			// Note that "i" represent the current algorithm's position in vectors "supp_density" and "supp_dimension"
			perf.start((*it).second);
			nLabels = (*it).first(binaryImg, labeledMat);
			perf.stop((*it).second);

			// BROAD RESULTS
			os << "\t" << perf.last((*it).second);
			write_n_labels ? os << "\t" << nLabels : os << "";
			// BROAD RESULTS

			// Add current time to "supp_density" and "supp_size" in the correct position 
			if (isdigit(filename[0]) && isdigit(filename[1]) && isdigit(filename[2])){
				// For density graph
				supp_density[i][ctoi(filename[1])].first += perf.last((*it).second);
				supp_density[i][ctoi(filename[1])].second++;

				// For dimension graph
				supp_size[i][ctoi(filename[0])].first += perf.last((*it).second);
				supp_size[i][ctoi(filename[0])].second++;
			}
			// Add current time to "supp_density" and "supp_size" in the correct position 

			if (output_colors){
                string out_folder = output_path + "\\" + output_folder + "\\" + colors_folder; 
                if (!dirExists(out_folder.c_str()))
                    if (0 != std::system(("mkdir " + out_folder).c_str()))
                        return ("Density_Size_Test on '" + input_folder + "': Unable to find/create the output path " + out_folder);

				normalizeLabels(labeledMat, nLabels);
				colorLabels(labeledMat, imgColors);
                imwrite(out_folder + "\\" + filename + "_" + (*it).second + ".png", imgColors);
			}
		}// END FOR
		os << endl;
	}// END WHILE

	// To calculate averages times
	vector<vector<long double>> density_averages(CCLAlgorithms.size(), vector<long double>(density)), size_averages(CCLAlgorithms.size(), vector<long double>(size));
	for (unsigned int i = 0; i < CCLAlgorithms.size(); ++i){
		// For all algorithms
		for (unsigned int j = 0; j < density_averages[i].size(); ++j){
			// For all density
			if (supp_density[i][j].second != 0)
				density_averages[i][j] = supp_density[i][j].first / supp_density[i][j].second;
			else
				density_averages[i][j] = 0.0;  // If there is no element with this density characyteristic the averages value is set to zero
		}
		for (unsigned int j = 0; j < size_averages[i].size(); ++j){
			// For all size
			if (supp_size[i][j].second != 0)
				size_averages[i][j] = supp_size[i][j].first / supp_size[i][j].second;
			else
				size_averages[i][j] = 0.0;  // If there is no element with this size characyteristic the averages value is set to zero
		}
	}
	// To calculate averages

	// To write density result on specified file
	for (unsigned int i = 0; i < density; ++i){
		// For every density
		if (density_averages[0][i] == 0.0) // Check it only for the first algorithm (it is the same for the others)
			density_os << "#"; // It means that there is no element with this density characyteristic 
		density_os << ((float)(i + 1) / 10) << "\t"; //Density value
		for (unsigned int j = 0; j < density_averages.size(); ++j){
			// For every alghorithm
			density_os << density_averages[j][i] << "\t";
		}
		density_os << endl; // End of current line (current density)
	}
	// To write density result on specified file

	// To set sizes's label 
	vector <pair<unsigned int, double>> supp_size_labels(size, make_pair(0, 0));

	// To write size result on specified file
	for (unsigned int i = 0; i < size; ++i){
		// For every size
		if (size_averages[0][i] == 0.0) // Check it only for the first algorithm (it is the same for the others)
			size_os << "#"; // It means that there is no element with this size characyteristic 
		supp_size_labels[i].first = (int)(pow(2, i + 5));
		supp_size_labels[i].second = size_averages[0][i];
		size_os << (int)pow(supp_size_labels[i].first,2) << "\t"; //Size value
		for (unsigned int j = 0; j < size_averages.size(); ++j){
			// For every alghorithms
			size_os << size_averages[j][i] << "\t";
		}
		size_os << endl; // End of current line (current size)
	}
	// To write size result

	// GNUPLOT SCRIPT
	string scriptos_path = output_path + "\\" + output_folder + "\\" + gnuplot_script;
	ofstream scriptos(scriptos_path);
	if (!scriptos.is_open())
		return ("Density_Size_Test on '" + input_folder + "': Unable to create " + scriptos_path);

    scriptos << "# This is a gnuplot (http://www.gnuplot.info/) script!" << endl;
    scriptos << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << gnuplot_script << "' if you want to run it" << endl << endl;
	
    scriptos << "reset" << endl; 
    scriptos << "cd '" << complete_output_path << "\'" << endl;
    scriptos << "set grid" << endl << endl; 

    // DENSITY
    scriptos << "# DENSITY GRAPH (COLORS)" << endl << endl; 

    scriptos << "set output \"" + output_density_graph + "\"" << endl;
    scriptos << "#set title \"Density\" font ', 12'" << endl << endl;

    scriptos << "# pdf colors" << endl; 
    scriptos << "set terminal pdf enhanced color font ',10'" << endl << endl;

    scriptos << "# Axes labels" << endl; 
    scriptos << "set xlabel \"Density\"" << endl;
    scriptos << "set ylabel \"Execution Time [ms]\"" << endl << endl;

    scriptos << "# Axes range" << endl;
    scriptos << "set xrange [0:1]" << endl;
    scriptos << "set yrange [*:*]" << endl;
    scriptos << "set logscale y" << endl << endl;

    scriptos << "# Legend" << endl;
    scriptos << "set key left top nobox spacing 2 font ', 8'" << endl << endl;

    scriptos << "# Plot" << endl;
	scriptos << "plot \\" << endl;
	vector<pair<CCLPointer, string>>::iterator it; // I need it after the cycle
	unsigned int i = 2;
	for (it = CCLAlgorithms.begin(); it != (CCLAlgorithms.end() - 1); ++it, ++i){
		scriptos << "\"" + output_density_result + "\" using 1:" << i << " with linespoints title \"" + (*it).second + "\" , \\" << endl;
	}
	scriptos << "\"" + output_density_result + "\" using 1:" << i << " with linespoints title \"" + (*it).second + "\"" << endl << endl;
	
    scriptos << "# DENSITY GRAPH (BLACK AND WHITE)" << endl << endl;
    
    scriptos << "set output \"" + output_density_graph_bw + "\"" << endl;
    scriptos << "#set title \"Density\" font ', 12'" << endl << endl;

    scriptos << "# pdf black and white" << endl;
    scriptos << "set terminal pdf enhanced monochrome dashed font ',10'" << endl << endl;

    scriptos << "replot" << endl << endl;

	// SIZE
    scriptos << "# SIZE GRAPH (COLORS)" << endl << endl;

    scriptos << "set output \"" + output_size_graph + "\"" << endl;
    scriptos << "set title \"Size\" font ',12'" << endl << endl;

    scriptos << "# pdf colors" << endl;
    scriptos << "set terminal pdf enhanced color font ',10'" << endl << endl;
    
    scriptos << "# Axes labels" << endl;
    scriptos << "set xlabel \"Pixels\"" << endl;
    scriptos << "set ylabel \"Execution Time [ms]\"" << endl << endl;

    scriptos << "# Axes range" << endl;
    scriptos << "set format x \"10^{%L}\"" << endl;
    scriptos << "set xrange [100:100000000]" << endl;
    scriptos << "set yrange [*:*]" << endl;
    scriptos << "set logscale xy 10" << endl << endl;

    scriptos << "# Legend" << endl;
    scriptos << "set key left top nobox spacing 2 font ', 8'" << endl;

    scriptos << "# Plot" << endl;
	//// Set Labels
	//for (unsigned int i=0; i < supp_size_labels.size(); ++i){
	//	if (supp_size_labels[i].second != 0){
	//		scriptos << "set label " << i+1 << " \"" << supp_size_labels[i].first << "x" << supp_size_labels[i].first << "\" at " << pow(supp_size_labels[i].first,2) << "," << supp_size_labels[i].second << endl;
	//	}
	//	else{
	//		// It means that there is no element with this size characyteristic so this label is not necessary
	//	}
	//}
	//// Set Labels
	scriptos << "plot \\" << endl;
	//vector<pair<CCLPointer, string>>::iterator it; // I need it after the cycle
	//unsigned int i = 2;
	i = 2;
	for (it = CCLAlgorithms.begin(); it != (CCLAlgorithms.end() - 1); ++it, ++i){
		scriptos << "\"" + output_size_result + "\" using 1:" << i << " with linespoints title \"" + (*it).second + "\" , \\" << endl;
	}
	scriptos << "\"" + output_size_result + "\" using 1:" << i << " with linespoints title \"" + (*it).second + "\"" << endl << endl;

    scriptos << "# SIZE (BLACK AND WHITE)" << endl << endl;

    scriptos << "set output \"" + output_size_graph_bw + "\"" << endl;
    scriptos << "#set title \"Size\" font ', 12'" << endl << endl;

    scriptos << "# pdf black and white" << endl;
    scriptos << "set terminal pdf enhanced monochrome dashed font ',10'" << endl << endl;

    scriptos << "replot" << endl << endl;

	scriptos << "exit gnuplot" << endl;

	density_os.close();
	size_os.close();
	scriptos.close();
	// GNUPLOT SCRIPT 

	if(0 != std::system(("gnuplot " + complete_output_path + "\\" + gnuplot_script).c_str()))
        return ("Density_Size_Test on '" + input_folder + "': Unable to run gnuplot's script");
	return ("Density_Size_Test on '" + output_folder + "': successfuly done");
}

int main(int argc, char **argv){

    ConfigFile cfg("config.cfg");  

    // Flags to customize output format
    bool output_colors_density_size = cfg.getValueOfKey<bool>("ds_colorLabels", false),
         output_colors_average_test = cfg.getValueOfKey<bool>("at_colorLabels", false),
         write_n_labels = cfg.getValueOfKey<bool>("write_n_labels", true),
         check_8connectivity = cfg.getValueOfKey<bool>("check_8connectivity", true),
         ds_saveMiddleTests = cfg.getValueOfKey<bool>("ds_saveMiddleTests", false),
         at_saveMiddleTests = cfg.getValueOfKey<bool>("at_saveMiddleTests", false);
    
    // Number of tests
    uint8_t ds_testsNumber = cfg.getValueOfKey<uint8_t>("ds_testsNumber", 1), 
            at_testsNumber = cfg.getValueOfKey<uint8_t>("at_testsNumber", 1);

	string input_txt = "files.txt",         /* Files who contains list of images's name on which CCLAlgorithms are tested */
           gnuplot_scipt = "gpScript.txt",  /* Name of gnuplot scripts*/
           colors_folder = "colors",
           middel_folder = "middle_results",
           output_path = cfg.getValueOfKey<string>("output_path", "output"), /* Folder on which result are stored */
           input_path = cfg.getValueOfKey<string>("input_path", "input");    /* Folder on which datasets are placed */
               
    // List of dataset on which CCLA are checked
    vector<string> check_list = cfg.getStringValuesOfKey("check_list", vector<string> {"test_random", "hamlet"});

    // Lists of dataset on which CCLA are tested: one list for every type of test
    vector<string> input_folders_density_size_test = cfg.getStringValuesOfKey("densitySize_tests" , vector<string> { "test_random"}),
				   input_folders_averages_test = cfg.getStringValuesOfKey("averages_tests" , vector<string> { "mirflickr", "tobacco800", "Sarc3D_masks", "hamlet" });

	// Check if algorithms are correct
    if (check_8connectivity){
        cout << "CHECK ALGORITHMS ON 8-CONNECTIVITY: " << endl;
        checkAlgorithms(CCLAlgorithms, check_list, input_path, input_txt);
    }
	// Check if algorithms are correct

	// Test Algorithms with different input type and different output format, and show execution result
	// AVERAGES TEST
	cout << endl << "AVERAGES TESTS: " << endl;
	for (unsigned int i = 0; i < input_folders_averages_test.size(); ++i){
		cout << "Averages_Test on '" << input_folders_averages_test[i] << "': starts" << endl;
		cout << averages_test(CCLAlgorithms, input_path, input_folders_averages_test[i], input_txt, gnuplot_scipt, output_path, colors_folder, write_n_labels, output_colors_average_test) << endl;
		cout << "Averages_Test on '" << input_folders_averages_test[i] << "': ends" << endl << endl;
	}
    
	// DENSITY_SIZE_TESTS
	cout << endl << "DENSITY_SIZE TESTS: " << endl;
	for (unsigned int i = 0; i < input_folders_density_size_test.size(); ++i){
		cout << "Density_Size_Test on '" << input_folders_density_size_test[i] << "': starts" << endl;
        cout << density_size_test(CCLAlgorithms, input_path, input_folders_density_size_test[i], input_txt, gnuplot_scipt, output_path, colors_folder, write_n_labels, output_colors_density_size) << endl;
		cout << "Density_Size_Test on '" << input_folders_density_size_test[i] << "': ends" << endl << endl;
	}

	return 0; 
}
	
