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

#ifndef YACCLAB_PROGRESS_BAR_H_
#define YACCLAB_PROGRESS_BAR_H_

#include <iostream>
#include <iomanip>
#include <string>

#include "system_info.h"

#define CONSOLE_WIDTH 80

/* This class is useful to display a progress bar in the output console.
Example of usage:
    progressBar p(number_of_things_to_do);
    p.start();

    //start cycle
        p.display(current_thing_number);
        //do something
    //end cycle

    p.end();
*/
class ProgressBar {
public:

    ProgressBar() {}

    ProgressBar(const size_t n_things_todo,
        const size_t gap = 4,
        const size_t console_width = 70,
        const std::string pre_message = "",
        const std::string post_message = "")
    {
        n_things_todo_ = n_things_todo;

        gap_ = gap;
        console_width_ = console_width;

        pre_message_ = pre_message;
        post_message_ = post_message;

        bar_width_ = console_width_ - pre_message_.size() - post_message_.size() - 5 /* 100%*/ - 2 /*[]*/;
    }

    void Start()
    {
        std::cout << pre_message_;
        std::cout << "[>";
        for (size_t i = 0; i < bar_width_ - 1; ++i) {
            std::cout << " ";
        }
        std::cout << "]   0%";
        std::cout << post_message_ << "\r";
        std::cout.flush();
        prev_ = 0;
    }

    void Display(const unsigned progress)
    {
        if (progress < n_things_todo_ && prev_ == (gap_ - 1)) {
            std::cout << pre_message_;
            std::cout << "[";
            size_t pos = bar_width_ * progress / n_things_todo_;
            for (size_t i = 0; i < bar_width_; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::string s = std::to_string(unsigned(progress * 100.0 / n_things_todo_));
            std::cout << "] " << std::string(3 - s.size(), ' ') << s << "%";
            std::cout << post_message_ << "\r";
            std::cout.flush();
            prev_ = 0;
        }
        else {
            prev_++;
        }
    }

    void End()
    {
        std::cout << pre_message_;
        std::cout << "[";
        for (size_t i = 0; i < bar_width_; ++i) {
            std::cout << "=";
        }
        std::cout << "] 100%";
        std::cout << post_message_;
        std::cout.flush();
        std::cout << std::endl;
        prev_ = 0;
    }

    ProgressBar(const size_t n_things_todo,
        const unsigned n_tests /*min 1 - max 999*/,
        const size_t gap = 4,
        const size_t console_width = 70,
        const std::string pre_message = "",
        const std::string post_message = "")
    {
        n_things_todo_ = n_things_todo;

        gap_ = gap;
        console_width_ = console_width;

        pre_message_ = pre_message;
        post_message_ = post_message;

        n_tests_ = n_tests;
        n_tests_length_ = std::to_string(n_tests).length();
        bar_width_ = console_width_ - pre_message_.size() - post_message_.size() - 5 /* 100%*/ - 2 /*[]*/ - 5 /*Test */ - (n_tests_length_*2 + 1) /*102/999*/ - 3 /* - */;
    }

    void StartRepeated()
    {
        std::cout << pre_message_ << "test " << std::setfill(' ') << std::setw(n_tests_length_) << cur_test_;
        std::cout << "/" << n_tests_ << " - ";
        std::cout << "[>";
        for (size_t i = 0; i < bar_width_ - 1; ++i) {
            std::cout << " ";
        }
        std::cout << "]   0%";
        std::cout << post_message_ << "\r";
        std::cout.flush();
        prev_ = 0;
    }

    void DisplayRepeated(const unsigned progress)
    {
        if (progress < n_things_todo_ && prev_ == (gap_ - 1)) {
            std::cout << pre_message_ << "test " << std::setfill(' ') << std::setw(n_tests_length_) << cur_test_;
            std::cout << "/" << n_tests_ << " - ";
            std::cout << "[";
            size_t pos = bar_width_ * progress / n_things_todo_;
            for (size_t i = 0; i < bar_width_; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::string s = std::to_string(unsigned(progress * 100.0 / n_things_todo_));
            std::cout << "] " << std::string(3 - s.size(), ' ') << s << "%";
            std::cout << post_message_ << "\r";
            std::cout.flush();
            prev_ = 0;
        }
        else {
            prev_++;
        }
    }

    bool EndRepeated()
    {
        std::cout << pre_message_ << "test " << std::setfill(' ') << std::setw(n_tests_length_) << cur_test_;
        std::cout << "/" << n_tests_ << " - ";
        std::cout << "[";
        for (size_t i = 0; i < bar_width_; ++i) {
            std::cout << "=";
        }
        std::cout << "] 100%";
        std::cout << post_message_;
        std::cout.flush();
        prev_ = 0;
        if (cur_test_ == n_tests_) {
            std::cout << std::endl;
            return true;
        }
        std::cout << "\r";
        cur_test_++; 
        return false;
}

private:
    size_t prev_;
    size_t n_things_todo_;
    size_t gap_;
    size_t bar_width_;
    size_t console_width_;
    std::string post_message_;
    std::string pre_message_;

    size_t n_tests_;
    size_t n_tests_length_;
    unsigned cur_test_ = 1;

};

/* This class is useful to display a title bar in the output console.
Example of usage:
    TODO
*/
class OutputBox {
public:

    OutputBox(const std::string& title = "", const size_t bar_width = CONSOLE_WIDTH, const size_t pre_spaces = 2)
    {
        pre_spaces_ = pre_spaces;
        bar_width_ = bar_width > CONSOLE_WIDTH ? CONSOLE_WIDTH : bar_width;
        bar_width_ -= pre_spaces_;

        if (title.size() > bar_width_ - 4) {
            title_ = title.substr(0, bar_width_ - 4 - 3) + "...";
        }
        else {
            title_ = title;
        }
        transform(title_.begin(), title_.end(), title_.begin(), ::toupper);

        if (title_ != "") {
            std::cout << "\n";
            PrintSeparatorLine();
            PrintData(title_);
            PrintSeparatorLine();
        }
    }

    void StartUnitaryBox(const std::string &dataset_name, const size_t n_things_todo)
    {
        PrintData(dataset_name + ":");
        std::string complete_pre_message = std::string(pre_spaces_, ' ') + "|  ";
        pb = ProgressBar(n_things_todo, 4, bar_width_ + pre_spaces_, complete_pre_message, " |");
        pb.Start();
    }

    void UpdateUnitaryBox(const size_t progress)
    {
        pb.Display(progress);
    }

    void StopUnitaryBox()
    {
        pb.End();
        PrintSeparatorLine();
    }

    void StartRepeatedBox(const std::string &dataset_name, const size_t n_things_todo, const size_t n_test_todo)
    {
        PrintData(dataset_name + ":");
        std::string complete_pre_message = std::string(pre_spaces_, ' ') + "|  ";
        pb = ProgressBar(n_things_todo, n_test_todo, 4, bar_width_ + pre_spaces_, complete_pre_message, " |");
        pb.StartRepeated();
    }

    void UpdateRepeatedBox(const size_t progress)
    {
        pb.DisplayRepeated(progress);
    }

    void StopRepeatedBox()
    {
        if (pb.EndRepeated()) {
            PrintSeparatorLine();
        }
    }

    void Cerror(const std::string& err, const std::string& title = "")
    {
        std::string complete_err = "";
        if (title != "") {
            PrintData(title + ":");
            complete_err = " ";
        }
        complete_err += "ERROR: [" + err + "]";
        PrintData(complete_err);
        PrintSeparatorLine();

        /*

        If the title is specified this function will print:

        | title:                                                                     |
        |  ERROR: [err]                                                              |
        +----------------------------------------------------------------------------+

        otherwise:

        |  ERROR: [err]                                                              |
        +----------------------------------------------------------------------------+

        */
    }

    void Cmessage(const std::string& msg)
    {
        std::string complete_msg = " MSG: [" + msg + "]";
        PrintData(complete_msg);

        /*

        This function will print:

        | MSG: [msg]                                                                  |

        */
    }

    void DisplayReport(const std::string &title, const std::vector<std::string> &messagges)
    {

        PrintData(title + ":");
        for (const auto& x : messagges) {
            PrintData(" " + x);
        }
        PrintSeparatorLine();
    }


private:

    void PrintSeparatorLine()
    {
        std::cout << std::string(pre_spaces_, ' ') << "+" << std::string(bar_width_ - 2, '-') << "+\n";
    }

    void PrintRawData(const std::string &data)
    {
        std::cout << std::string(pre_spaces_, ' ') << "| " << data << std::string(bar_width_ - data.size() - 4, ' ') << " |\n";
    }

    void PrintData(const std::string &data)
    {
        unsigned step = bar_width_ - 4;
        std::string tab = "    ";
        for (unsigned i = 0; i < data.length(); i += step) {
            if (i == 0) {
                PrintRawData(data.substr(i, step));
                i += tab.size();
                step -= tab.size();
            }
            else {
                PrintRawData(tab + data.substr(i, step));
            }
        }
    }

    size_t bar_width_;
    std::string title_;
    size_t pre_spaces_;
    ProgressBar pb;
};

#endif // !YACCLAB_PROGRESS_BAR_H_

/*

+------------------------------------------------------------------------------+
| Checking Algorithms on 8-Connectivity                                        |
+------------------------------------------------------------------------------+
| 3dpes:                                                                       |
| [=====================================================================] 100% |
+------------------------------------------------------------------------------+
| tobacco800:                                                                  |
| [=====================================================================] 100% |
+------------------------------------------------------------------------------+
| medical:                                                                     |
| [=====================================================================] 100% |
+------------------------------------------------------------------------------+
| hamlet:                                                                      |
| [error]: questo è un messaggio molto molto molto ma molto lungo di prova per |
|      vedere come fare se non ci stiamo									   |
| [============================================>                        ]  69% |







*/