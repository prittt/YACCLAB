// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_PERFORMANCE_EVALUATOR_H_
#define YACCLAB_PERFORMANCE_EVALUATOR_H_

#include <map>

#include "opencv2/core.hpp"

class PerformanceEvaluator {
    struct Elapsed {
        double last;
        double total;

        Elapsed() : last(0), total(0) {}
    };

public:
    PerformanceEvaluator()
    {
        tick_frequency_ = cv::getTickFrequency();
    }

    void start()
    {
        int64 ticks = cv::getTickCount();
        counter_.last = static_cast<double>(ticks);
    }

    double stop()
    {
        int64 ticks = cv::getTickCount();
        double t = ticks - counter_.last;
        counter_.last = t;
        counter_.total += t;
        return counter_.last*1000. / tick_frequency_;
    }

    void reset()
    {
        counter_.total = 0;
    }

    double last()
    {
        return counter_.last*1000. / tick_frequency_;
    }

    double total()
    {
        return counter_.total*1000. / tick_frequency_;
    }

    void store(const std::string& s, double time /*milliseconds*/)
    {
        counters_[s].last = time;
        counters_[s].total += time;
    }

    double get(const std::string& s)
    {
        return counters_.at(s).last;
    }

    bool find(const std::string& s)
    {
        return counters_.find(s) != counters_.end();
    }

private:
    double tick_frequency_;
    Elapsed counter_;
    std::map<std::string, Elapsed> counters_;
};

#endif // !YACCLAB_PERFORMANCE_EVALUATOR_H_