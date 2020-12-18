// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "file_manager.h"

#include <algorithm>
#include <fstream>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>

#include "system_info.h"

using namespace std;

const char filesystem::path::separator_ =
#ifdef YACCLAB_WINDOWS
'\\';
#else
'/';
#endif

bool filesystem::exists(const path& p)
{
	struct stat info;
	string s = p.string();
	const char* path_to_check = s.c_str();
	if (stat(path_to_check, &info) != 0) {
		//printf("cannot access %s\n", pathname);
		return false;
	}
	else if (info.st_mode & S_IFDIR) {
		//printf("%s is a directory\n", pathname);
		return true; // is directory
	}

	//printf("%s is no directory\n", pathname);
	return true; // is file
}

bool filesystem::exists(const path& p, bool &is_dir)
{
	struct stat info;
    string s = p.string();
    const char* path_to_check = s.c_str();
    if (stat(path_to_check, &info) != 0) {
        //printf("cannot access %s\n", pathname);
        return false;
    }
    else if (info.st_mode & S_IFDIR) {
        //printf("%s is a directory\n", pathname);
		is_dir = true;
        return true; // is directory
    }

    //printf("%s is no directory\n", pathname);
	is_dir = false;
    return true; // is file
}

bool filesystem::exists(const path& p, error_code& ec)
{
    return filesystem::exists(p);
}

bool filesystem::exists(const path& p, error_code& ec, bool& is_dir)
{
	return filesystem::exists(p, is_dir);
}

bool filesystem::create_directories(const path& p)
{
    string s(p.string());
    string parameters = "";
#if defined(YACCLAB_UNIX) || defined(YACCLAB_LINUX) || defined(YACCLAB_APPLE)
    // make it recursive by adding "-p" suffix
    parameters = "-p";
#endif

    if (!exists(s)) {
        if (system(("mkdir " + parameters + " \"" + s + "\"").c_str()) != 0) {
            //cerr << "Unable to find/create the output path " + s;
            return false;
        }
    }
    return true;
}

bool filesystem::create_directories(const path& p, error_code& ec)
{
    return filesystem::create_directories(p);
}

void filesystem::path::NormalizePath()
{
#if defined(YACCLAB_UNIX) || defined(YACCLAB_LINUX) || defined(YACCLAB_APPLE)
    std::replace(path_.begin(), path_.end(), '\\', '/');
#elif defined(YACCLAB_WINDOWS)
    std::replace(path_.begin(), path_.end(), '/', '\\');
#endif

    return;
}

void filesystem::copy(const path& from, const path& to)
{
    if (!filesystem::exists(from)) {
        return;
    }

    if (!filesystem::exists(to)) {
        if (filesystem::create_directories(to.parent_path())) {
            ifstream src(from.string());
            ofstream dst(to.string());

            dst << src.rdbuf();
        }
    }
}

void filesystem::copy(const path& from, const path& to, error_code& ec)
{
    filesystem::copy(from, to);
}