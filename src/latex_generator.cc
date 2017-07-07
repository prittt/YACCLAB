#include "latex_generator.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "utilities.h"
#include "system_info.h"

using namespace cv;
using namespace std;

// To generate latex table with average results
void GenerateLatexTable(const string& output_path, const string& latex_file, const Mat1d& all_res, const vector<String>& datasets_name, const vector<String>& ccl_algorithms)
{
    string latex_path = output_path + kPathSeparator + latex_file;
    ofstream is(latex_path);
    if (!is.is_open()) {
        cout << "Unable to open/create " + latex_path << endl;
        return;
    }

    // fixed number of decimal values
    is << fixed;
    is << setprecision(3);

    is << "%This table format needs the package 'siunitx', please uncomment and add the following line code in latex preamble if you want to add the table in your latex file" << endl;
    is << "%\\usepackage{siunitx}" << endl << endl;
    is << "\\begin{table}[tbh]" << endl << endl;
    is << "\t\\centering" << endl;
    is << "\t\\caption{Average Results in ms (Lower is Better)}" << endl;
    is << "\t\\label{tab:table1}" << endl;
    is << "\t\\begin{tabular}{|l|";
    for (unsigned i = 0; i < ccl_algorithms.size(); ++i)
        is << "S[table-format=2.3]|";
    is << "}" << endl;
    is << "\t\\hline" << endl;
    is << "\t";
    for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
        //EraseDoubleEscape(datasets_name);
        //datasets_name.erase(std::remove(datasets_name.begin(), datasets_name.end(), '\\'), datasets_name.end());
        is << " & {" << ccl_algorithms[i] << "}"; //Header
    }
    is << "\\\\" << endl;
    is << "\t\\hline" << endl;

    for (unsigned i = 0; i < datasets_name.size(); ++i) {
        is << "\t" << datasets_name[i];
        for (int j = 0; j < all_res.cols; ++j) {
            is << " & ";
            if (all_res(i, j) != numeric_limits<double>::max())
                is << all_res(i, j); //Data
        }
        is << "\\\\" << endl;
    }
    is << "\t\\hline" << endl;
    is << "\t\\end{tabular}" << endl << endl;
    is << "\\end{table}" << endl;

    is.close();
}

// To generate latex table with memory average accesses
void GenerateMemoryLatexTable(const string& output_path, const string& latex_file, const Mat1d& accesses, const string& dataset, const vector<String>& ccl_mem_algorithms)
{
    // TODO handle if folder does not exists
    string latex_path = output_path + kPathSeparator + dataset + kPathSeparator + latex_file;
    ofstream is(latex_path);
    if (!is.is_open()) {
        cout << "Unable to open/create " + latex_path << endl;
        return;
    }

    // fixed number of decimal values
    is << fixed;
    is << setprecision(3);

    is << "%This table format needs the package 'siunitx', please uncomment and add the following line code in latex preamble if you want to add the table in your latex file" << endl;
    is << "%\\usepackage{siunitx}" << endl << endl;
    is << "\\begin{table}[tbh]" << endl << endl;
    is << "\t\\centering" << endl;
    is << "\t\\caption{Analysis of memory accesses required by connected components computation for '" << dataset << "' dataset. The numbers are given in millions of accesses}" << endl;
    is << "\t\\label{tab:table1}" << endl;
    is << "\t\\begin{tabular}{|l|";
    for (int i = 0; i < accesses.cols + 1; ++i)
        is << "S[table-format=2.3]|";
    is << "}" << endl;
    is << "\t\\hline" << endl;
    is << "\t";

    // Header
    is << "{Algorithm} & {Binary Image} & {Label Image} & {Equivalence Vector/s}  & {Other} & {Total Accesses}";
    is << "\\\\" << endl;
    is << "\t\\hline" << endl;

    for (unsigned i = 0; i < ccl_mem_algorithms.size(); ++i) {
        // For every algorithm
        const String& alg_name = ccl_mem_algorithms[i];
        //EraseDoubleEscape(alg_name);
        is << "\t{" << alg_name << "}";

        double tot = 0;

        for (int s = 0; s < accesses.cols; ++s) {
            // For every data_ structure
            if (accesses(i, s) != 0)
                is << "\t& " << (accesses(i, s) / 1000000);
            else
                is << "\t& ";

            tot += (accesses(i, s) / 1000000);
        }
        // Total Accesses
        is << "\t& " << tot;

        // EndLine
        is << "\t\\\\" << endl;
    }

    // EndTable
    is << "\t\\hline" << endl;
    is << "\t\\end{tabular}" << endl << endl;
    is << "\\end{table}" << endl;

    is.close();
}

void GenerateLatexCharts(const string& output_path, const string& latex_charts, const string& latex_folder, const vector<String>& datasets_name)
{
    string latex_path = output_path + kPathSeparator + latex_folder + kPathSeparator + latex_charts;
    ofstream is(latex_path);
    if (!is.is_open()) {
        cout << "Unable to open/create " + latex_path << endl;
        return;
    }

    SystemInfo info;
    string chartSize{ "0.45" }, chartWidth{ "1" };

    // Get information about date and time
    string datetime = GetDatetime();

    // fixed number of decimal values
    is << fixed;
    is << setprecision(3);

    is << "%These charts are generated using YACCLAB. Follow our project on GitHub: https://github.com/prittt/YACCLAB" << endl << endl;
    is << "\\begin{figure}[b]" << endl << endl;
    //\newcommand{ \machineName }{x86\_MSVC2015\_Windows10}
    is << "\t\\newcommand{\\machineName}{";
    is << info << "}" << endl;

    //replace the . with _ for filenames
    pair<string, string> compiler(SystemInfo::GetCompiler());
    replace(compiler.first.begin(), compiler.first.end(), '.', '_');
    replace(compiler.second.begin(), compiler.second.end(), '.', '_');

    //\newcommand{ \compilerName }{MSVC2015}
    is << "\t\\newcommand{\\compilerName}{" + compiler.first + compiler.second + "}" << endl;

    is << "\t\\centering" << endl;

    for (unsigned i = 0; i < datasets_name.size(); ++i) {
        is << "\t\\begin{subfigure}[b]{" + chartSize + "\\textwidth}" << endl;
        is << "\t\t\\caption{" << datasets_name[i] + "}" << endl;
        is << "\t\t\\centering" << endl;
        is << "\t\t\\includegraphics[width=" + chartWidth + "\\textwidth]{\\compilerName " + datasets_name[i] + ".pdf}" << endl;
        is << "\t\\end{subfigure}" << endl << endl;
    }

    is << "\t\\caption{\\machineName}\t" + datetime << endl << endl;
    is << "\\end{figure}" << endl << endl;

    is.close();
}