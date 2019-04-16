// Copyright(c) 2016 - 2019 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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

#include "latex_generator.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "utilities.h"
#include "file_manager.h"
#include "system_info.h"

using namespace cv;
using namespace std;
using namespace filesystem;

//// To generate latex table with average results
//void GenerateLatexTable(const path& output_path, const string& latex_file, const Mat1d& all_res, const vector<String>& datasets_name, const vector<String>& ccl_algorithms)
//{
//    //string latex_path = output_path + kPathSeparator + latex_file;
//    path latex_path = output_path / path(latex_file);
//    ofstream os(latex_path.string());
//    if (!os.is_open()) {
//        dmux::cout << "Unable to open/create " + latex_path.string() << endl;
//        return;
//    }
//
//    // fixed number of decimal values
//    os << fixed;
//    os << setprecision(3);
//
//    os << "%This table format needs the package 'siunitx', please uncomment and add the following line code in latex preamble if you want to add the table in your latex file" << endl;
//    os << "%\\usepackage{siunitx}" << endl << endl;
//    os << "\\begin{table}[tbh]" << endl << endl;
//    os << "\t\\centering" << endl;
//    os << "\t\\caption{Average Results in ms (Lower os Better)}" << endl;
//    os << "\t\\label{tab:table1}" << endl;
//    os << "\t\\begin{tabular}{|l|";
//    for (unsigned i = 0; i < ccl_algorithms.size(); ++i)
//        os << "S[table-format=2.3]|";
//    os << "}" << endl;
//    os << "\t\\hline" << endl;
//    os << "\t";
//    for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
//        //RemoveCharacter(datasets_name, '\\');
//        //datasets_name.erase(std::remove(datasets_name.begin(), datasets_name.end(), '\\'), datasets_name.end());
//        os << " & {" << ccl_algorithms[i] << "}"; //Header
//    }
//    os << "\\\\" << endl;
//    os << "\t\\hline" << endl;
//
//    for (unsigned i = 0; i < datasets_name.size(); ++i) {
//        os << "\t" << datasets_name[i];
//        for (int j = 0; j < all_res.cols; ++j) {
//            os << " & ";
//            if (all_res(i, j) != numeric_limits<double>::max())
//                os << all_res(i, j); //Data
//        }
//        os << "\\\\" << endl;
//    }
//    os << "\t\\hline" << endl;
//    os << "\t\\end{tabular}" << endl << endl;
//    os << "\\end{table}" << endl;
//
//    os.close();
//}
//
//// To generate latex table with memory average accesses
//void GenerateMemoryLatexTable(const path& output_path, const string& latex_file, const Mat1d& accesses, const string& dataset, const vector<String>& ccl_mem_algorithms)
//{
//    // TODO handle if folder does not exists
//    //string latex_path = output_path + kPathSeparator + dataset + kPathSeparator + latex_file;
//    path latex_path = output_path / path(dataset) / path(latex_file);
//    ofstream os(latex_path.string());
//    if (!os.is_open()) {
//        dmux::cout << "Unable to open/create " + latex_path.string() << endl;
//        return;
//    }
//
//    // fixed number of decimal values
//    os << fixed;
//    os << setprecision(3);
//
//    os << "%This table format needs the package 'siunitx', please uncomment and add the following line code in latex preamble if you want to add the table in your latex file" << endl;
//    os << "%\\usepackage{siunitx}" << endl << endl;
//    os << "\\begin{table}[tbh]" << endl << endl;
//    os << "\t\\centering" << endl;
//    os << "\t\\caption{Analysis of memory accesses required by connected components computation for '" << dataset << "' dataset. The numbers are given in millions of accesses}" << endl;
//    os << "\t\\label{tab:table1}" << endl;
//    os << "\t\\begin{tabular}{|l|";
//    for (int i = 0; i < accesses.cols + 1; ++i)
//        os << "S[table-format=2.3]|";
//    os << "}" << endl;
//    os << "\t\\hline" << endl;
//    os << "\t";
//
//    // Header
//    os << "{Algorithm} & {Binary Image} & {Label Image} & {Equivalence Vector/s}  & {Other} & {Total Accesses}";
//    os << "\\\\" << endl;
//    os << "\t\\hline" << endl;
//
//    for (unsigned i = 0; i < ccl_mem_algorithms.size(); ++i) {
//        // For every algorithm
//        const String& alg_name = ccl_mem_algorithms[i];
//        //RemoveCharacter(alg_name, '\\');
//        os << "\t{" << alg_name << "}";
//
//        double tot = 0;
//
//        for (int s = 0; s < accesses.cols; ++s) {
//            // For every data_ structure
//            if (accesses(i, s) != 0)
//                os << "\t& " << (accesses(i, s) / 1000000);
//            else
//                os << "\t& ";
//
//            tot += (accesses(i, s) / 1000000);
//        }
//        // Total Accesses
//        os << "\t& " << tot;
//
//        // EndLine
//        os << "\t\\\\" << endl;
//    }
//
//    // EndTable
//    os << "\t\\hline" << endl;
//    os << "\t\\end{tabular}" << endl << endl;
//    os << "\\end{table}" << endl;
//
//    os.close();
//}
//
//void GenerateLatexCharts(const filesystem::path& output_path, const string& latex_charts, const string& latex_folder, const vector<String>& datasets_name)
//{
//    //string latex_path = output_path + kPathSeparator + latex_folder + kPathSeparator + latex_charts;
//    path latex_path = output_path / path(latex_folder) / path(latex_charts);
//    ofstream os(latex_path.string());
//    if (!os.is_open()) {
//        dmux::cout << "Unable to open/create " + latex_path.string() << endl;
//        return;
//    }
//
//    string chart_size{ "0.45" }, chart_width{ "1" };
//
//    // Get information about date and time
//    string datetime = GetDatetime();
//
//    // fixed number of decimal values
//    os << fixed;
//    os << setprecision(3);
//
//    os << "%These charts are generated using YACCLAB. Follow our project on GitHub: https://github.com/prittt/YACCLAB" << endl << endl;
//    os << "\\begin{figure}[b]" << endl << endl;
//    //\newcommand{ \machineName }{x86\_MSVC2015\_Windows10}
//    os << "\t\\newcommand{\\machineName}{";
//    //os << info << "}" << endl; ------------------------------------------------------------------------------------
//
//    SystemInfo s_info(cfg);
//    //replace the . with _ for filenames
//    string compiler_name(s_info.compiler_name());
//    string compiler_version(s_info.compiler_version());
//    std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');
//
//    //\newcommand{ \compilerName }{MSVC2015}
//    os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << endl;
//
//    os << "\t\\centering" << endl;
//
//    for (unsigned i = 0; i < datasets_name.size(); ++i) {
//        os << "\t\\begin{subfigure}[b]{" + chart_size + "\\textwidth}" << endl;
//        os << "\t\t\\caption{" << datasets_name[i] + "}" << endl;
//        os << "\t\t\\centering" << endl;
//        os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName " + datasets_name[i] + ".pdf}" << endl;
//        os << "\t\\end{subfigure}" << endl << endl;
//    }
//
//    os << "\t\\caption{\\machineName}\t" + datetime << endl << endl;
//    os << "\\end{figure}" << endl << endl;
//
//    os.close();
//}

//void LatexGenerator(const map<string, bool>& test_to_perform, const path& latex_path, const string& latex_file, const Mat1d& all_res, const vector<String>& datasets_name, const vector<String>& ccl_algorithms, const vector<String>& ccl_mem_algorithms, const map<string, Mat1d>& accesses, ConfigData& cfg)
//{
//    path latex = latex_path / path(latex_file);
//    ofstream os(latex.string());
//    if (!os.is_open()) {
//        dmux::cout << "Unable to open/create " + latex.string() << endl;
//        return;
//    }
//
//    // fixed number of decimal values
//    os << fixed;
//    os << setprecision(3);
//
//    // Document begin
//    os << "%These file is generated by YACCLAB. Follow our project on GitHub: https://github.com/prittt/YACCLAB" << endl << endl;
//    os << "\\documentclass{article}" << endl << endl;
//
//    os << "\\usepackage{siunitx}" << endl;
//    os << "\\usepackage{graphicx}" << endl;
//    os << "\\usepackage{subcaption}" << endl << endl;
//    os << "\\begin{document}" << endl << endl;
//
//    // Section average results table ------------------------------------------------------------------------------------------
//    if (test_to_perform.at("perform_average")) {
//        os << "\\section{Average Table Results}" << endl << endl;
//
//        os << "\\begin{table}[tbh]" << endl << endl;
//        os << "\t\\centering" << endl;
//        os << "\t\\caption{Average Results in ms (Lower is better)}" << endl;
//        os << "\t\\label{tab:table1}" << endl;
//        os << "\t\\begin{tabular}{|l|";
//        for (unsigned i = 0; i < ccl_algorithms.size(); ++i)
//            os << "S[table-format=2.3]|";
//        os << "}" << endl;
//        os << "\t\\hline" << endl;
//        os << "\t";
//        for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
//            //RemoveCharacter(datasets_name, '\\');
//            //datasets_name.erase(std::remove(datasets_name.begin(), datasets_name.end(), '\\'), datasets_name.end());
//            os << " & {" << EscapeUnderscore(ccl_algorithms[i]) << "}"; //Header
//        }
//        os << "\\\\" << endl;
//        os << "\t\\hline" << endl;
//        for (unsigned i = 0; i < datasets_name.size(); ++i) {
//            os << "\t" << datasets_name[i];
//            for (int j = 0; j < all_res.cols; ++j) {
//                os << " & ";
//                if (all_res(i, j) != numeric_limits<double>::max())
//                    os << all_res(i, j); //Data
//            }
//            os << "\\\\" << endl;
//        }
//        os << "\t\\hline" << endl;
//        os << "\t\\end{tabular}" << endl << endl;
//        os << "\\end{table}" << endl;
//    }
//
//    // Section figures with charts ---------------------------------------------------------------------------
//    if (test_to_perform.at("perform_average") || test_to_perform.at("perform_density")) {
//        SystemInfo s_info(cfg);
//        string info_to_latex = s_info.build() + "_" + s_info.compiler_name() + s_info.compiler_version() + "_" + s_info.os();
//        std::replace(info_to_latex.begin(), info_to_latex.end(), ' ', '_');
//        info_to_latex = EscapeUnderscore(info_to_latex);
//
//        string chart_size{ "0.45" }, chart_width{ "1" };
//        // Get information about date and time
//        string datetime = GetDatetime();
//
//        os << "\\section{Average and Density Charts}" << endl << endl;
//
//        os << "\\begin{figure}[b]" << endl << endl;
//        //\newcommand{ \machineName }{x86\_MSVC2015\_Windows10}
//        os << "\t\\newcommand{\\machineName}{";
//        os << info_to_latex << "}" << endl;
//
//        string compiler_name(s_info.compiler_name());
//        string compiler_version(s_info.compiler_version());
//        //replace the . with _ for compiler strings
//        std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');
//
//        //\newcommand{ \compilerName }{MSVC2015}
//        os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << endl;
//
//        os << "\t\\centering" << endl;
//
//        for (unsigned i = 0; i < datasets_name.size(); ++i) {
//            os << "\t\\begin{subfigure}[b]{" + chart_size + "\\textwidth}" << endl;
//            os << "\t\t\\caption{" << datasets_name[i] + "}" << endl;
//            os << "\t\t\\centering" << endl;
//            os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName " + datasets_name[i] + ".pdf}" << endl;
//            os << "\t\\end{subfigure}" << endl << endl;
//        }
//
//        os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << endl << endl;
//        os << "\\end{figure}" << endl << endl;
//    }
//
//    // Section memory result table ---------------------------------------------------------------------------
//    if (test_to_perform.at("perform_memory")) {
//        os << "\\section{Memory Accesses tests}" << endl << endl;
//
//        for (const auto& dataset : accesses) {
//            const auto& dataset_name = dataset.first;
//            const auto& accesses = dataset.second;
//
//            os << "\\begin{table}[tbh]" << endl << endl;
//            os << "\t\\centering" << endl;
//            os << "\t\\caption{Analysis of memory accesses required by connected components computation for '" << dataset_name << "' dataset. The numbers are given in millions of accesses}" << endl;
//            os << "\t\\label{tab:table1}" << endl;
//            os << "\t\\begin{tabular}{|l|";
//            for (int i = 0; i < accesses.cols + 1; ++i)
//                os << "S[table-format=2.3]|";
//            os << "}" << endl;
//            os << "\t\\hline" << endl;
//            os << "\t";
//
//            // Header
//            os << "{Algorithm} & {Binary Image} & {Label Image} & {Equivalence Vector/s}  & {Other} & {Total Accesses}";
//            os << "\\\\" << endl;
//            os << "\t\\hline" << endl;
//
//            for (unsigned i = 0; i < ccl_mem_algorithms.size(); ++i) {
//                // For every algorithm
//                const String& alg_name = ccl_mem_algorithms[i];
//                //RemoveCharacter(alg_name, '\\');
//                os << "\t{" << alg_name << "}";
//
//                double tot = 0;
//
//                for (int s = 0; s < accesses.cols; ++s) {
//                    // For every data_ structure
//                    if (accesses(i, s) != 0)
//                        os << "\t& " << (accesses(i, s) / 1000000);
//                    else
//                        os << "\t& ";
//
//                    tot += (accesses(i, s) / 1000000);
//                }
//                // Total Accesses
//                os << "\t& " << tot;
//
//                // EndLine
//                os << "\t\\\\" << endl;
//            }
//
//            // EndTable
//            os << "\t\\hline" << endl;
//            os << "\t\\end{tabular}" << endl << endl;
//            os << "\\end{table}" << endl;
//        }
//    }
//
//    os << "\\end{document}";
//    os.close();
//}