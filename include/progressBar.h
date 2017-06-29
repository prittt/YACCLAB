// Copyright(c) 2016 - 2017 Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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

#pragma once
#include <iostream>
#include "systemInfo.h"

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
    ProgressBar(const size_t total, const size_t gap = 4, const size_t barWidth = 70)
    {
        total_ = total;
        gap_ = gap;
        barWidth_ = barWidth;
    }

    void Start()
    {
        std::cout << "[>";
        for (size_t i = 0; i < barWidth_ - 1; ++i)
        {
            std::cout << " ";
        }
        std::cout << "] 0 %\r";
        std::cout.flush();
        prev_ = 0;
    }

    void Display(const size_t progress, const int currentRepeat = -1)
    {
        if (progress < total_ && prev_ == (gap_ - 1))
        {
            if (currentRepeat > 0)
            {
                std::cout << "Test #" << currentRepeat << " ";
            }
            std::cout << "[";
            size_t pos = barWidth_ * progress / total_;
            for (size_t i = 0; i < barWidth_; ++i)
            {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << size_t(progress * 100.0 / total_) << " %\r";
            std::cout.flush();
            prev_ = 0;
        }
        else
        {
            prev_++;
        }
    }

    void End(const int lastRepeat = -1)
    {
        if (lastRepeat > 0)
        {
            std::cout << "Test #" << lastRepeat << " ";
        }
        std::cout << "[";
        for (size_t i = 0; i < barWidth_; ++i)
        {
            std::cout << "=";
        }
        std::cout << "] 100 %\r";
        std::cout.flush();
        prev_ = 0;
        std::cout << std::endl;
    }

private:
    size_t prev_;
    size_t total_;
    size_t gap_;
    size_t barWidth_;
};

/* This class is useful to display a title bar in the output console.
Example of usage:
    titleBar t("NAME");
    t.start();

        // do something

    t.end();
*/
class TitleBar {
public:
    TitleBar(const std::string& title, const size_t barWidth = 70, const size_t asterisks = 15)
    {
        title_ = title;
        barWidth_ = barWidth;
        asterisks_ = asterisks;

        if ((asterisks_ * 2 + title.size() + 8) > barWidth)
            barWidth_ = asterisks_ * 2 + title.size() + 8;
    }

    void Start()
    {
        size_t spaces = size_t((barWidth_ - (title_.size() + 6 + asterisks_ * 2)) / 2);
        PrintAsterisks();
        PrintSpaces(spaces);
        std::cout << title_ << " starts ";
        PrintSpaces(spaces);
        PrintAsterisks();
        std::cout << std::endl;
    }

    void End()
    {
        size_t spaces = size_t((barWidth_ - (title_.size() + 5 + asterisks_ * 2)) / 2);
        PrintAsterisks();
        PrintSpaces(spaces);
        std::cout << title_ << " ends ";
        PrintSpaces(spaces);
        PrintAsterisks();
        std::cout << std::endl << std::endl;
    }

private:

    void PrintAsterisks()
    {
        for (size_t i = 0; i < asterisks_; ++i)
        {
            std::cout << "*";
        }
    }

    void PrintSpaces(size_t spaces)
    {
        for (size_t i = 0; i < spaces; ++i)
        {
            std::cout << " ";
        }
    }

    std::string title_;
    size_t barWidth_;
    size_t asterisks_;
};
