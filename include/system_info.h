// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_SYSTEM_INFO_H_
#define YACCLAB_SYSTEM_INFO_H_

#include <iostream>
#include <string>

#include <config_data.h>

#if defined YACCLAB_WITH_CUDA
#include "cuda_runtime.h"
#endif

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

// extern struct ConfigData cfg;

/*@brief Retrieve system information

Singleton class that retrieves machine information like the CPU
brand name, the OS used, and the architecture employed.

*/
class SystemInfo {
public:
    static SystemInfo &GetInstance();

    // Return the brand and model of the CPU used
    static std::string cpu() { return GetInstance().cpu_; }

    // Return the architecture (x86 or x64) used
    static std::string build() { return GetInstance().build_; }

    // Return the Operating System used
    static std::string os() { return GetInstance().os_ + " " + GetInstance().os_bit_; }

    // Return the compiler_ used (name and version)
    static std::string compiler_name() { return GetInstance().compiler_name_; }
    static std::string compiler_version() { return GetInstance().compiler_version_; }

    static void set_os(std::string os) { GetInstance().SetOs(os); }

    SystemInfo(SystemInfo const&) = delete;
    void operator=(SystemInfo const&) = delete;

private:
    SystemInfo()
    {
        SetBuild();
        SetCpuBrand();
        SetOsBit();
        SetCompiler();
    }

    std::string cpu_;
    std::string build_;
    std::string os_;
    std::string os_bit_;
    std::string compiler_name_;
    std::string compiler_version_;

    void SetCpuBrand();
    void SetBuild();
    void SetOs(std::string os);
    void SetOsBit();
    void SetCompiler();
};



#if defined YACCLAB_WITH_CUDA
struct CudaInfo {
public: 
    static CudaInfo& GetInstance();

    static std::string device_name() { return GetInstance().device_name_; }
    static std::string cuda_capability() { return GetInstance().cuda_capability_; }
    static std::string runtime_version() { return GetInstance().runtime_version_; }
    static std::string driver_version() { return GetInstance().driver_version_; }

    CudaInfo(CudaInfo const&) = delete;
    void operator=(CudaInfo const&) = delete;

private:
    std::string CudaBeautifyVersionNumber(int v);

    CudaInfo();

    std::string device_name_;
    std::string cuda_capability_;
    std::string runtime_version_;
    std::string driver_version_;
};
#endif

#endif // !YACCLAB_SYSTEM_INFO_H_