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

# include "foldersManager.h"

using namespace std; 

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

bool dirExists(const string& pathname){
	struct stat info;
	const char* path = pathname.c_str(); 
	if (stat(path, &info) != 0){
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

bool makeDir(const string& path){
	if (!dirExists(path.c_str())){
		if (0 != std::system(("mkdir " + path).c_str())){
			cout << "Unable to find/create the output path " + path;
			return false;
		}
	}
	return true;
}

void listFiles(const char* dir, vector<string>& filesList, const int fileTypes, const string extension)
{
	char originalDirectory[_MAX_PATH];

	// Get the current directory so we can return to it
	_getcwd(originalDirectory, _MAX_PATH);

	// Change to the working directory
	_chdir(dir);  
	_finddata_t fileinfo;

	// This will grab the first file in the directory
	// "*" can be changed if you only want to look for specific files
	intptr_t handle = _findfirst(("*" + extension).c_str(), &fileinfo);

	// No files or directories found
	if (handle == -1)  
		return; 

	do
	{
		if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
			continue; // skip . and .. files
		if (fileinfo.attrib & _A_SUBDIR) // Use bitmask to see if this is a directory
		{	
			// This is a directory
			if (fileTypes == FM_ONLY_SUBDIR || fileTypes == FM_BOTH)
				filesList.push_back(fileinfo.name); 
		}
		else
		{
			// This is a file
			if (fileTypes == FM_ONLY_FILES || fileTypes == FM_BOTH)
				filesList.push_back(fileinfo.name);
		}
			
	} while (_findnext(handle, &fileinfo) == 0);

	_findclose(handle); // Close the stream

	_chdir(originalDirectory);
}

void listFiles(const string& dir, vector<string>& filesList, const int fileTypes, const string extension)
{
	const char* _dir = dir.c_str(); 

	char originalDirectory[_MAX_PATH];

	// Get the current directory so we can return to it
	_getcwd(originalDirectory, _MAX_PATH);

	// Change to the working directory
	_chdir(_dir);
	_finddata_t fileinfo;

	// This will grab the first file in the directory
	// "*" can be changed if you only want to look for specific files
	intptr_t handle = _findfirst(("*" + extension).c_str(), &fileinfo);

	// No files or directories found
	if (handle == -1)
		return;

	do
	{
		if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
			continue; // skip . and .. files
		if (fileinfo.attrib & _A_SUBDIR) // Use bitmask to see if this is a directory
		{
			// This is a directory
			if (fileTypes == FM_ONLY_SUBDIR || fileTypes == FM_BOTH)
				filesList.push_back(fileinfo.name);
		}
		else
		{
			// This is a file
			if (fileTypes == FM_ONLY_FILES || fileTypes == FM_BOTH)
				filesList.push_back(fileinfo.name);
		}

	} while (_findnext(handle, &fileinfo) == 0);

	_findclose(handle); // Close the stream

	_chdir(originalDirectory);
}

bool fileExists(const string& path) {
	ifstream file(path.c_str());
	return file.good();
}