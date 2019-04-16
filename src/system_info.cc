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

#include "config_data.h"
#include "system_info.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

SystemInfo& SystemInfo::GetInstance()
{
    static SystemInfo instance;
    return instance;
}

void SystemInfo::SetCpuBrand()
{
    cpu_ = "cpu_unknown";

#if defined(_MSC_VER) && defined(YACCLAB_WINDOWS)
    // Compiler dependent, works on Windows and Visual Studio
    int cpu_info[4] = { -1 };
    unsigned nExIds, i = 0;
    char cpu_name[0x40];
    // Get the information associated with each extended ID.
    __cpuid(cpu_info, 0x80000000);
    nExIds = cpu_info[0];
    for (i = 0x80000000; i <= nExIds; ++i) {
        __cpuid(cpu_info, i);
        // Interpret CPU brand string
        if (i == 0x80000002)
            memcpy(cpu_name, cpu_info, sizeof(cpu_info));
        else if (i == 0x80000003)
            memcpy(cpu_name + 16, cpu_info, sizeof(cpu_info));
        else if (i == 0x80000004)
            memcpy(cpu_name + 32, cpu_info, sizeof(cpu_info));
    }
    cpu_ = { cpu_name };
#elif defined(YACCLAB_WINDOWS)
    // Compiler independent, works on Windows
    std::system("wmic cpu get name|more > cpu_name.txt");
    ifstream is("cpu_name.txt");
    if (!is.is_open()) {
        return;
    }
    string line;
    std::getline(is, line);
    std::getline(is, line);
    is.close();
    cpu_ = line;
#elif defined(YACCLAB_LINUX) || defined(YACCLAB_UNIX)
    ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        cpu_ = "cpu_unknown";
    }
    string cpu_name;
    while (getline(cpuinfo, cpu_name)) {
        if (cpu_name.substr(0, cpu_name.find(":") - 1) == "model name") {
            cpu_name = cpu_name.substr(cpu_name.find(":") + 2);
            break;
        }
    }
    cpuinfo.close();
    cpu_ = cpu_name;
#elif defined(YACCLAB_APPLE)
#define BUFFERLEN 100
    char buffer[BUFFERLEN];
    size_t bufferlen = BUFFERLEN;
    sysctlbyname("machdep.cpu.brand_string", &buffer, &bufferlen, NULL, 0);
    cpu_ = string(buffer); // AppleClang doesn't like braces initializer list of c++11
#endif

    const char* t = " \t\n\r\f\v";
    // Remove heading and trailing special characters in cpu string
    cpu_.erase(0, cpu_.find_first_not_of(t));
    cpu_.erase(cpu_.find_last_not_of(t) + 1);

    // Delete "CPU" characters, if present, from cpu name
    const string pattern = " CPU";
    string::size_type n = pattern.length();
    for (string::size_type i = cpu_.find(pattern); i != string::npos; i = cpu_.find(pattern)) {
        cpu_.erase(i, n);
    }
}

void SystemInfo::SetBuild()
{
    if (sizeof(void*) == 4)
        build_ = "x86";
    else if (sizeof(void*) == 8)
        build_ = "x64";
    else
        build_ = "build_unknown";
}

void SystemInfo::SetOsBit()
{
    os_bit_ = "";

#if defined(YACCLAB_WINDOWS)
    // Compiler independent, works on Windows
    typedef BOOL(WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
    LPFN_ISWOW64PROCESS fnIsWow64Process;
    BOOL bIsWow64 = FALSE;

    //IsWow64Process is not available on all supported versions of Windows.
    //Use GetModuleHandle to get a handle to the DLL that contains the function
    //and GetProcAddress to get a pointer to the function if available.

    fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(
        GetModuleHandle(TEXT("kernel32")), "IsWow64Process");

    if (NULL != fnIsWow64Process) {
        if (!fnIsWow64Process(GetCurrentProcess(), &bIsWow64)) {
            //handle error
            return;
        }
    }
    if (bIsWow64)
        os_bit_ = "64 bit";
    else
        os_bit_ = "32 bit";

#elif defined(YACCLAB_LINUX) || defined(YACCLAB_UNIX)
    struct utsname unameData;
    uname(&unameData);
    string bit = unameData.machine;
    if (bit == "x86_64")
        os_bit_ = "64 bit";
    else if (bit == "i686")
        os_bit_ = "32 bit";
#elif defined(YACCLAB_APPLE)
    // TODO
#endif
}

void SystemInfo::SetOs(std::string os)
{
    os_ = os;
}

void SystemInfo::SetCompiler()
{
    compiler_name_ = "compiler_unknown";
    compiler_version_ = "";

#if defined(__clang__)
    /* Clang/LLVM. ---------------------------------------------- */
    compiler_name_ = "Clang";
    compiler_version_ = to_string(__clang_major__) + " " + to_string(__clang_minor__);

#elif defined(__ICC) || defined(__INTEL_COMPILER)
    /* Intel ICC/ICPC. ------------------------------------------ */
    compiler_name_ = "IntelC++";
    compiler_version_ = to_string(__INTEL_COMPILER);

#elif defined(__GNUC__) || defined(__GNUG__)
    /* GNU GCC/G++. --------------------------------------------- */
    compiler_name_ = "GCC";

#ifdef __VERSION__
    compiler_version_ = string(__VERSION__);
#endif // __VERSION__

#elif defined(__HP_cc) || defined(__HP_aCC)
    /* Hewlett-Packard C/C++. ---------------------------------- */
    compiler_name_ = "HP_C++";

#elif defined(__IBMC__) || defined(__IBMCPP__)
    /* IBM XL C/C++. -------------------------------------------- */
    compiler_name_ = "IBM_XL_C-C++";

#elif defined(_MSC_VER)
    /* Microsoft Visual Studio. --------------------------------- */
    compiler_name_ = "VS";
    if (_MSC_VER >= 1910) //Visual Studio 2017, MSVC++ 15.0
        compiler_version_ = "15.0";
    else if (_MSC_VER == 1900)
        compiler_version_ = "14.0";
    else if (_MSC_VER == 1800)
        compiler_version_ = "12.0";
    else if (_MSC_VER == 1700)
        compiler_version_ = "11.0";
    else if (_MSC_VER == 1600)
        compiler_version_ = "10.0";
    else if (_MSC_VER == 1500)
        compiler_version_ = "9.0";
    else if (_MSC_VER == 1400)
        compiler_version_ = "8.0";
    else if (_MSC_VER == 1310)
        compiler_version_ = "7.1";
    else if (_MSC_VER == 1300)
        compiler_version_ = "7.0";
    else if (_MSC_VER == 1200)
        compiler_version_ = "6.0";

#elif defined(__PGI)
    /* Portland Group PGCC/PGCPP. ------------------------------- */
    compiler_name_ = "PGCC-PGCPP";

#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
    /* Oracle Solaris Studio. ----------------------------------- */
    compiler_name_ = "SUNPRO";

#endif
}

#if defined USE_CUDA
CudaInfo& CudaInfo::GetInstance()
{
    static CudaInfo instance;
    return instance;
}

std::string CudaInfo::CudaBeautifyVersionNumber(int v) {
    int minor = (v / 10) % 10;
    int major = v / 1000;
    return std::to_string(major) + '.' + std::to_string(minor);
}

CudaInfo::CudaInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int runtimeVersion, driverVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    device_name_ = std::string(prop.name);
    cuda_capability_ = std::to_string(prop.major) + '.' + std::to_string(prop.minor);
    runtime_version_ = CudaBeautifyVersionNumber(runtimeVersion);
    driver_version_ = CudaBeautifyVersionNumber(driverVersion);
}
#endif