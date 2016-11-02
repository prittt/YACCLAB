// Copyright(c) 2016 - Costantino Grana, Federico Bolelli, Lorenzo Baraldi and Roberto Vezzani
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
#include "opencv2/opencv.hpp"

enum memorydatatype{

	// Data structures for "classical" algorithms 
	MD_BINARY_MAT = 0,
	MD_LABELED_MAT = 1,
	MD_EQUIVALENCE_VEC = 2,

	// Other data structures
	MD_OTHER = 3,

	// Total number of data structures in the list
	MD_SIZE = 4, 
};


template <typename T>
class memMat {
public:

	int rows; 
	int cols; 

	memMat(cv::Mat_<T> img){
		_img = img.clone(); // Deep copy
		_accesses = cv::Mat1i(img.size(), 0); 
		rows = img.rows; 
		cols = img.cols; 
	}

	memMat(cv::Size size){
		_img = cv::Mat_<T>(size); 
		_accesses = cv::Mat1i(size, 0);
		rows = size.height; 
		cols = size.width; 
	}

	memMat(cv::Size size, const T val){
		_img = cv::Mat_<T>(size, val);
		_accesses = cv::Mat1i(size, 1);	// The initilization accesses must be counted
		rows = size.height;
		cols = size.width;
	}

	T& operator()(const int r, const int c) {
		_accesses.ptr<int>(r)[c]++; // Count access
		return _img.template ptr<T>(r)[c];
	}

	cv::Mat_<T> getImage() const{
		return _img.clone(); 
	}

	cv::Mat1i getAcessesMat() const{
		return _accesses.clone(); 
	}

	double getTotalAcesses() const{
		return cv::sum(_accesses)[0]; 
	}

	//~memMat();  // This is the destructor: declaration

private:
	cv::Mat_<T> _img;
	cv::Mat1i _accesses;
};

template <typename T>
class memVector {
public:
	memVector(std::vector<T> vec){
		_vec = vec;  // Deep copy
		_accesses = std::vector<int>(vec.size(), 0);
	}

	memVector(const size_t size){
		_vec = std::vector<T>(size);
		_accesses = std::vector<int>(size, 0);
	}

	memVector(const size_t size, const T val){
		_vec = std::vector<T>(size, val);
		_accesses = std::vector<int>(size, 1); // The initilization accesses must be counted
	}

	T& operator[](const int i){
		_accesses[i]++; // Count access
		return _vec[i];
	}

	std::vector<T> getVector() const{
		return _vec;
	}

	std::vector<T> getAcessesVector() const{
		return _accesses;
	}

	double getTotalAcesses() const{
		double tot = 0; 
		for (size_t i = 0; i < _accesses.size(); ++i){
			tot += _accesses[i]; 
		}

		return tot; 
	}

	T* getDataPointer(){
		return _vec.data();
	}

	size_t size(){
		return _vec.size();
	}

	void memiota(size_t begin, size_t end, const T value){

		T _value = value;

		for (size_t i = begin; i < end; ++i){
			_vec[i] = _value++;
			_accesses[i]++;	// increment access
		}
	}

	//~memVector();  // This is the destructor: declaration

private:
	std::vector<T> _vec;
	std::vector<int> _accesses;
};

//template <typename T>
//class memVector {
//public:
//	memVector(std::vector<T> vec){
//		_vec = vec;  // Deep copy
//		_accesses = cv::Mat1i(1, vec.size(), 0);
//	}
//
//	memVector(const size_t size){
//		_vec = vector<T>(size); 
//		_accesses = cv::Mat1i(1, size, 0);
//	}
//
//	memVector(const size_t size, const T val){
//		_vec = vector<T>(size, val);
//		_accesses = cv::Mat1i(1, size, 1); // The initilization accesses must be counted
//	}
//
//	T& operator[](const int i){
//		_accesses.ptr<int>(0)[i]++; // Count access
//		return _vec[i];
//	}
//
//	std::vector<T> getVector() const{
//		return _vec;
//	}
//
//	cv::Mat1i getAcessesVector() const{
//		return _accesses.clone();
//	}
//
//	double getTotalAcesses() const{
//		return cv::sum(_accesses)[0];
//	}
//
//	T* getDataPointer(){
//		return _vec.data(); 
//	}
//
//	size_t size(){
//		return _vec.size(); 
//	}
//
//	void memiota(size_t begin, size_t end, const T value){
//		
//		T _value = value; 
//		
//		for (size_t i = begin; i < end; ++i){
//			_vec[i] = _value++; 
//			_accesses(0,i)++;	// increment access
//		}
//	}
//
//	//~memVector();  // This is the destructor: declaration
//
//private:
//	std::vector<T> _vec;
//	cv::Mat1i _accesses;
//};