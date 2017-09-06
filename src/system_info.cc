// Copyright(c) 2016 - 2017 Federico Bolelli, Costantino Grana, Michele Cancilla, Lorenzo Baraldi and Roberto Vezzani
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
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

SystemInfo::SystemInfo(ConfigData& cfg)
{
    SetBuild();
    SetCpuBrand();
    SetOs(cfg);
    SetCompiler();
}

void SystemInfo::SetCpuBrand()
{
#ifdef WINDOWS
    cpu_ = GetWindowsCpu();
#elif defined(LINUX) || defined(UNIX)
    cpu_ = GetLinuxCpu();
#elif defined(APPLE)
    cpu_ = GetAppleCpu();
#else
    cpu_ = "cpu_unknown";
#endif

    const char* t = " \t\n\r\f\v";

    // Remove heading and trailing spaces in string
    cpu_.erase(0, cpu_.find_first_not_of(t));
    cpu_.erase(cpu_.find_last_not_of(t) + 1);

    // Delete "CPU" characters from CPUBrandString, if present
    const string pattern = " CPU";
    string::size_type n = pattern.length();
    for (string::size_type i = cpu_.find(pattern); i != string::npos; i = cpu_.find(pattern))
        cpu_.erase(i, n);
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

#if defined(WINDOWS)
string SystemInfo::GetWindowsCpu()
{
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

    return { cpu_name };
}
//
//bool SystemInfo::GetWinMajorMinorVersion(DWORD& major, DWORD& minor)
//{
//    bool bRetCode = false;
//    LPBYTE pinfoRawData = 0;
//    if (NERR_Success == NetWkstaGetInfo(NULL, 100, &pinfoRawData)) {
//        WKSTA_INFO_100* pworkstationInfo = (WKSTA_INFO_100*)pinfoRawData;
//        major = pworkstationInfo->wki100_ver_major;
//        minor = pworkstationInfo->wki100_ver_minor;
//        ::NetApiBufferFree(pinfoRawData);
//        bRetCode = true;
//    }
//    return bRetCode;
//}
//
//string SystemInfo::GetWindowsVersion()
//{
//    string winver;
//    OSVERSIONINFOEX osver;
//    SYSTEM_INFO sysInfo;
//    typedef void(__stdcall *GETSYSTEMINFO) (LPSYSTEM_INFO);
//
//    __pragma(warning(push))
//        __pragma(warning(disable:4996))
//        memset(&osver, 0, sizeof(osver));
//    osver.dwOSVersionInfoSize = sizeof(osver);
//    GetVersionEx((LPOSVERSIONINFO)&osver);
//    __pragma(warning(pop))
//        DWORD major = 0;
//    DWORD minor = 0;
//    if (GetWinMajorMinorVersion(major, minor)) {
//        osver.dwMajorVersion = major;
//        osver.dwMinorVersion = minor;
//    }
//    else if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 2) {
//        OSVERSIONINFOEXW osvi;
//        ULONGLONG cm = 0;
//        cm = VerSetConditionMask(cm, VER_MINORVERSION, VER_EQUAL);
//        ZeroMemory(&osvi, sizeof(osvi));
//        osvi.dwOSVersionInfoSize = sizeof(osvi);
//        osvi.dwMinorVersion = 3;
//        if (VerifyVersionInfoW(&osvi, VER_MINORVERSION, cm)) {
//            osver.dwMinorVersion = 3;
//        }
//    }
//
//    GETSYSTEMINFO getSysInfo = (GETSYSTEMINFO)GetProcAddress(GetModuleHandle((LPCTSTR)"kernel32.dll"), "GetNativeSystemInfo");
//    if (getSysInfo == NULL)  getSysInfo = ::GetSystemInfo;
//    getSysInfo(&sysInfo);
//
//    if (osver.dwMajorVersion == 10 && osver.dwMinorVersion >= 0 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows 10 Server";
//    if (osver.dwMajorVersion == 10 && osver.dwMinorVersion >= 0 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 10";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 3 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2012 R2";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 3 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 8.1";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 2 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2012";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 2 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 8";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 1 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2008 R2";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 1 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 7";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 0 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2008";
//    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 0 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows Vista";
//    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 2 && osver.wProductType == VER_NT_WORKSTATION
//        &&  sysInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64)  winver = "Windows XP";
//    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 2)   winver = "Windows Server 2003";
//    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 1)   winver = "Windows XP";
//    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 0)   winver = "Windows 2000";
//    if (osver.dwMajorVersion < 5)   winver = "unknown";
//
//    if (osver.wServicePackMajor != 0) {
//        std::string sp;
//        char buf[128] = { 0 };
//        sp = " Service Pack ";
//        sprintf_s(buf, sizeof(buf), "%hd", osver.wServicePackMajor);
//        sp.append(buf);
//        winver += sp;
//    }
//
//    typedef BOOL(WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
//    LPFN_ISWOW64PROCESS fnIsWow64Process;
//    BOOL bIsWow64 = FALSE;
//
//    fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(
//        GetModuleHandle(TEXT("kernel32")), "IsWow64Process");
//
//    if (NULL != fnIsWow64Process) {
//        if (!fnIsWow64Process(GetCurrentProcess(), &bIsWow64)) {
//            //handle error
//        }
//    }
//
//    if ((bIsWow64 && build_ == "x86") || (build_ == "x64")) {
//        //64 bit OS and 32 bit built application or 64 bit application
//        winver += " 64 bit";
//    }
//    else if ((!bIsWow64 && build_ == "x86")) {
//        //32 bit OS and 32 bit built application
//        winver += " 32 bit";
//    }
//
//    return winver;
//}

#elif defined(LINUX) || defined(UNIX) 
string SystemInfo::GetLinuxCpu()
{
    ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return "cpu_unknown";
    }
    string cpu_name;
    while (getline(cpuinfo, cpu_name)) {
        if (cpu_name.substr(0, cpu_name.find(":") - 1) == "model name") {
            cpu_name = cpu_name.substr(cpu_name.find(":") + 2);
            break;
        }
    }
    cpuinfo.close();
    return cpu_name;
}

//string GetLinuxOs()
//{
//    struct utsname unameData;
//    uname(&unameData);
//
//    string bit = unameData.machine;
//
//    if (bit == "x86_64")
//        bit = "64 bit";
//    else if (bit == "i686")
//        bit = "32 bit";
//    else
//        bit = "bit_unknown";
//
//    return string(unameData.sysname) + " " + bit;
//}
#elif defined(APPLE)
string SystemInfo::GetAppleCpu()
{
    // https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man3/sysctl.3.html#//apple_ref/doc/man/3/sysctl
    // http://stackoverflow.com/questions/853798/programmatically-get-processor-details-from-mac-os-x
    // http://osxdaily.com/2011/07/15/get-cpu-info-via-command-line-in-mac-os-x/

    string cpu_name = system("sysctl - n machdep.cpu.brand_string");
    return cpu_name;
}

#endif

void SystemInfo::SetOs(ConfigData& cfg)
{
//#if defined(WINDOWS)
//    os_ = GetWindowsVersion();
//#elif defined(UNIX) || defined(LINUX)
//    os_ = GetLinuxOs();
//#elif defined(APPLE)
//    os_ = "Mac OSX";
//#else
//    os_ = "os_unknown";
//#endif

    os_ = cfg.yacclab_os;
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
    compiler_version_ = string(__INTEL_COMPILER);

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

     // else "None"

#elif defined(__PGI)
    /* Portland Group PGCC/PGCPP. ------------------------------- */
    compiler_name_ = "PGCC-PGCPP";

#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
    /* Oracle Solaris Studio. ----------------------------------- */
    compiler_name_ = "SUNPRO";

#endif
}