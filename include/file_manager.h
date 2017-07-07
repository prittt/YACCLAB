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

#ifndef YACCLAB_FILE_MANAGER_H_
#define YACCLAB_FILE_MANAGER_H_

#include <string>

/** @brief Check if folder exist or not

 This function simply check if folder of specified pathname exists or not. If the folder
 exists and it could be accessed the function return true, otherwise false.

@param[in] pathname path of the folder to check

@return true if the specified folder exists and could be accessed, false otherwise

*/
bool DirExists(const char* pathname);

/** @overload

This function simply check if folder of specified pathname exists or not. If the folder
exists and it could be accessed the function return true, otherwise false.

@param[in] pathName path of the folder to check

@return true if the specified folder exists and could be accessed, false otherwise

*/
bool DirExists(const std::string& pathName);

/** @brief Creates a directory if it does not exist

 This function checks if directory exist and creates it if not. It returns true if the
 directory exists or creation process end correctly, false otherwise

@param[in] path path of the folder to create

@return true if the specified folder exists and could be accessed, false otherwise

*/
bool MakeDir(const std::string& path);

/** @brief Check if file exist or not

This function simply check if file of specified pathname exists or not. If the file
exists and it could be accessed the function return true, otherwise false.

@param[in] pathname path of the file to check

@return true if the specified file exists and could be accessed, false otherwise

*/
bool FileExists(const std::string& path);

/** @brief Standardize path

This function standardize path with '\' if YACCLAB is run on Windows OSs
or with '/' if it is run on Unix OSs.

@param[in] pathname path to normalize

@return string of the normalized path

*/
std::string NormalizePath(const std::string& path);

#endif // !YACCLAB_FILE_MANAGER_H_