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

#ifndef YACCLAB_SYSTEM_INFO_H_
#define YACCLAB_SYSTEM_INFO_H_

#include <iostream>
#include <string>

#include <config_data.h>

#if _WIN32 || _WIN64 || WIN32 || __WIN32__ || __WINDOWS__ || __TOS_WIN__
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifndef NOMINMAX
#define NOMINMAX // Prevent <Windows.h> header file defines its own macros named max and min
#endif
#include <WINDOWS.h>
#include <lm.h>
#pragma comment(lib, "netapi32.lib")
#define YACCLAB_WINDOWS
#elif  __gnu_linux__ || __linux__
#define YACCLAB_LINUX
#include <sys/utsname.h>
#elif  __unix || __unix__
#define YACCLAB_UNIX
#include <sys/utsname.h>
#elif __APPLE__ || __MACH__ || macintosh || Macintosh || (__APPLE__ && __MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#define YACCLAB_APPLE
#endif

extern struct ConfigData cfg;

/*@brief Retrieve system information

Class that retrieves machine information like the CPU
brand name, the OS used, and the architecture employed.

*/
class SystemInfo {
public:
    SystemInfo(ConfigData& cfg)
    {
        SetBuild();
        SetCpuBrand();
        SetOs(cfg);
        SetOsBit();
        SetCompiler();
    }

    // Return the brand and model of the CPU used
    std::string cpu() { return cpu_; }

    // Return the architecture (x86 or x64) used
    std::string build() { return build_; }

    // Return the Operating System used
    std::string os() { return os_ + " " + os_bit_; }

    // Return the compiler_ used (name and version)
    std::string compiler_name() { return compiler_name_; }
    std::string compiler_version() { return compiler_version_; }

private:
    std::string cpu_;
    std::string build_;
    std::string os_;
    std::string os_bit_;
    std::string compiler_name_;
    std::string compiler_version_;

    void SetCpuBrand();
    void SetBuild();
    void SetOs(ConfigData& cfg);
    void SetOsBit();
    void SetCompiler();
};

#endif // !YACCLAB_SYSTEM_INFO_H_