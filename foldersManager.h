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

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>
#include <direct.h>
#include <cstring>
#include <io.h>
#include <iostream>
#include <stdio.h>


/* File types for the "listFiles" function */
enum fileTypes {
	FM_ONLY_SUBDIR = 0,
	FM_ONLY_FILES = 1,
	FM_BOTH = 2
};

/** @brief Check if folder exist or not

 This function simply check if folder of specified pathname exists or not. If the folder
 exists and it could be accesed the function return true, otherwise false.

@param[in] pathname path of the folder to check

@return true if the specified folder exists and could be accesed, false otherwise

*/
bool dirExists(const char* pathname); 

/** @overload

This function simply check if folder of specified pathname exists or not. If the folder
exists and it could be accesed the function return true, otherwise false.

@param[in] pathname path of the folder to check

@return true if the specified folder exists and could be accesed, false otherwise

*/
bool dirExists(const std::string& pathname);

/** @brief Creates a directory if it does not exist

 This function check if directory exist and creates it if not. It returns true if the 
 directory exists or creation process end correctly, false otherwise

@param[in] path path of the folder to create

@return true if the specified folder exists and could be accesed, false otherwise

*/
bool makeDir(const std::string& path); 

/** @brief Get the list of files placed in the specified directory

This function get the list of files placed in a specific directory.  
The list of file names are save in a string vector. Note that "." and ".." are not 
considered.

@param[in] dir directory on which search the list of files
@param[out] fileList string vector containing the file names list
@param[in] fileTypes integer parameter to specify what king of "files" search. Allowed 
			values are: FM_ONLY_SUBDIR (0) to search only subdirectories, FM_ONLY_FILES (1)
			to search only file and FM_BOTH (2) to search both subdirectories and files
@param[in] extension specifies the extension of files to search. The parameter is optional. 
			For example, to get the list of file names with extension "png" in directory "dir"
			set extension parameter to ".png"

@return nothing
*/
void listFiles(const char* dir, std::vector<std::string>& filesList, const int fileTypes = FM_BOTH, const std::string extension = "");


/** @overload

This function get the list of files placed in a specific directory.
The list of file names are save in a string vector. Note that "." and ".." are not
considered.

@param[in] dir directory on which search the list of files
@param[out] fileList string vector containing the file names list
@param[in] fileTypes integer parameter to specify what king of "files" search. Allowed
values are: FM_ONLY_SUBDIR (0) to search only subdirectories, FM_ONLY_FILES (1)
to search only file and FM_BOTH (2) to search both subdirectories and files
@param[in] extension specifies the extension of files to search. The parameter is optional.
For example, to get the list of file names with extension "png" in directory "dir"
set extension parameter to ".png"

@return nothing
*/
void listFiles(const std::string& dir, std::vector<std::string>& filesList, const int fileTypes = FM_BOTH, const std::string extension = "");

/** @brief Check if file exist or not

This function simply check if file of specified pathname exists or not. If the file
exists and it could be accesed the function return true, otherwise false.

@param[in] pathname path of the file to check

@return true if the specified file exists and could be accesed, false otherwise

*/
bool fileExists(const std::string& path);
