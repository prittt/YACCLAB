#pragma once
#include "opencv2/opencv.hpp"

class PerformanceEvaluator {
	struct elapsed {
		double _last;
		double _total;

		elapsed() : _last(0),_total(0) {}
	};

public:
	PerformanceEvaluator() {
		_tickFrequency = cv::getTickFrequency();
	}
	void start() {
		_counter._last = (double)cv::getTickCount();
	}
	double stop() {
		double t = cv::getTickCount() - _counter._last;
		_counter._last = t;
		_counter._total += t;
		return _counter._last*1000./_tickFrequency;
	}
	void reset() {
		_counter._total = 0;
	}
	double last() {
		return _counter._last*1000./_tickFrequency;
	}
	double total() {
		return _counter._total*1000./_tickFrequency;
	}

	void start (const std::string& s) {
		_counters[s]._last = (double)cv::getTickCount();
	}
	double stop (const std::string& s) {
		elapsed& e = _counters[s];
		double t = cv::getTickCount() - e._last;
		e._last = t;
		e._total += t;
		return e._last*1000./_tickFrequency;
	}
	void reset (const std::string& s) {
		_counters[s]._total = 0;
	}
	double last (const std::string& s) {
		return _counters[s]._last*1000./_tickFrequency;
	}
	double total (const std::string& s) {
		return _counters[s]._total*1000./_tickFrequency;
	}

private:
	double _tickFrequency;
	elapsed _counter;
	std::map<std::string,elapsed> _counters;
};