#include "systemInfo.h"

using namespace std;

systemInfo::systemInfo()
{
    this->cpuBrand = getCpuBrand();
    this->build = getBuild();
    this->os = getOs();
    this->compiler = getCompiler();
}

string systemInfo::getCpuBrand()
{
#if defined(WINDOWS)
    return getWindowsCpuBrand();
#elif defined(LINUX) || defined(UNIX)
    return getLinuxCpuBrand();
#elif defined(APPLE)
    return getAppleCpuBrand();
#else
    return "CpuUnkwonwn";
#endif
}

string systemInfo::getBuild()
{
    string build;

    // Check WINDOWS
//#if defined(WINDOWS)
//#if _WIN64
//#define ENVIRONMENT64
//    architecture = "x64";
//#else
//#define ENVIRONMENT32
//    architecture = "x86";
//
//#endif
//    // Check Linux
//#elif defined(LINUX) || defined(UNIX)
//
//#if __x86_64__ || __ppc64__
//#define ENVIRONMENT64
//    architecture = "x64";
//#else
//#define ENVIRONMENT32
//    architecture = "x86";
//#endif
//
//#elif defined(APPLE)
//        //TODO
//#else
//    architecture = "ArchitectureUnkwonwn";
//#endif
    if (sizeof(void*) == 4)
        build = "x86";
    else if (sizeof(void*) == 8)
        build = "x64";
    else
        build = "Unkwonwn";

    return build;
}

#if  defined(WINDOWS)
string systemInfo::getWindowsCpuBrand()
{
    int CPUInfo[4] = { -1 };
    unsigned nExIds, i = 0;
    char CPUBrandString[0x40];
    // Get the information associated with each extended ID.
    __cpuid(CPUInfo, 0x80000000);
    nExIds = CPUInfo[0];
    for (i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(CPUInfo, i);
        // Interpret CPU brand string
        if (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    //string includes manufacturer, model and clockspeed
    string CPU(CPUBrandString);

    //Remove leading and trailing spaces
    CPU = trim(CPU);

    return CPU;
}

bool systemInfo::GetWinMajorMinorVersion(DWORD& major, DWORD& minor)
{
    bool bRetCode = false;
    LPBYTE pinfoRawData = 0;
    if (NERR_Success == NetWkstaGetInfo(NULL, 100, &pinfoRawData))
    {
        WKSTA_INFO_100* pworkstationInfo = (WKSTA_INFO_100*)pinfoRawData;
        major = pworkstationInfo->wki100_ver_major;
        minor = pworkstationInfo->wki100_ver_minor;
        ::NetApiBufferFree(pinfoRawData);
        bRetCode = true;
    }
    return bRetCode;
}

string systemInfo::GetWindowsVersion()
{
    string winver;
    OSVERSIONINFOEX osver;
    SYSTEM_INFO sysInfo;
    typedef void(__stdcall *GETSYSTEMINFO) (LPSYSTEM_INFO);

    __pragma(warning(push))
        __pragma(warning(disable:4996))
        memset(&osver, 0, sizeof(osver));
    osver.dwOSVersionInfoSize = sizeof(osver);
    GetVersionEx((LPOSVERSIONINFO)&osver);
    __pragma(warning(pop))
        DWORD major = 0;
    DWORD minor = 0;
    if (GetWinMajorMinorVersion(major, minor))
    {
        osver.dwMajorVersion = major;
        osver.dwMinorVersion = minor;
    }
    else if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 2)
    {
        OSVERSIONINFOEXW osvi;
        ULONGLONG cm = 0;
        cm = VerSetConditionMask(cm, VER_MINORVERSION, VER_EQUAL);
        ZeroMemory(&osvi, sizeof(osvi));
        osvi.dwOSVersionInfoSize = sizeof(osvi);
        osvi.dwMinorVersion = 3;
        if (VerifyVersionInfoW(&osvi, VER_MINORVERSION, cm))
        {
            osver.dwMinorVersion = 3;
        }
    }

    GETSYSTEMINFO getSysInfo = (GETSYSTEMINFO)GetProcAddress(GetModuleHandle((LPCTSTR)"kernel32.dll"), "GetNativeSystemInfo");
    if (getSysInfo == NULL)  getSysInfo = ::GetSystemInfo;
    getSysInfo(&sysInfo);

    if (osver.dwMajorVersion == 10 && osver.dwMinorVersion >= 0 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows 10 Server";
    if (osver.dwMajorVersion == 10 && osver.dwMinorVersion >= 0 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 10";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 3 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2012 R2";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 3 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 8.1";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 2 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2012";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 2 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 8";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 1 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2008 R2";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 1 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows 7";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 0 && osver.wProductType != VER_NT_WORKSTATION)  winver = "Windows Server 2008";
    if (osver.dwMajorVersion == 6 && osver.dwMinorVersion == 0 && osver.wProductType == VER_NT_WORKSTATION)  winver = "Windows Vista";
    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 2 && osver.wProductType == VER_NT_WORKSTATION
        &&  sysInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64)  winver = "Windows XP";
    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 2)   winver = "Windows Server 2003";
    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 1)   winver = "Windows XP";
    if (osver.dwMajorVersion == 5 && osver.dwMinorVersion == 0)   winver = "Windows 2000";
    if (osver.dwMajorVersion < 5)   winver = "unknown";

    if (osver.wServicePackMajor != 0)
    {
        std::string sp;
        char buf[128] = { 0 };
        sp = " Service Pack ";
        sprintf_s(buf, sizeof(buf), "%hd", osver.wServicePackMajor);
        sp.append(buf);
        winver += sp;
    }

    typedef BOOL(WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
    LPFN_ISWOW64PROCESS fnIsWow64Process;
    BOOL bIsWow64 = FALSE;

    fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(
        GetModuleHandle(TEXT("kernel32")), "IsWow64Process");

    if (NULL != fnIsWow64Process)
    {
        if (!fnIsWow64Process(GetCurrentProcess(), &bIsWow64))
        {
            //handle error
        }
    }

    if ((bIsWow64 && getBuild() == "x86") || (getBuild() == "x64"))
    {
        //64 bit OS and 32 bit built application or 64 bit application
        winver += " 64 bit";
    }
    else if ((!bIsWow64 && getBuild() == "x86"))
    {
        //32 bit OS and 32 bit built application
        winver += " 32 bit";
    }

    return winver;
}

#elif defined(LINUX) || defined(UNIX)
string systemInfo::getLinuxCpuBrand()
{
    ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open())
    {
        cout << "errore di apertura del file\n";
        return "CpuUnkwonwn";
    }
    string CPUBrandString;
    while (getline(cpuinfo, CPUBrandString))
    {
        if (CPUBrandString.substr(0, CPUBrandString.find(":") - 1) == "model name")
        {
            CPUBrandString = CPUBrandString.substr(CPUBrandString.find(":") + 2);
            break;
        }
    }
    //Remove leading and trailing spaces
    CPUBrandString = trim(CPUBrandString);

    //If delete "CPU" characters from CPUBrandString, if present
    const string pattern = " CPU";
    string::size_type n = pattern.length();
    for (string::size_type i = CPUBrandString.find(pattern); i != string::npos; i = CPUBrandString.find(pattern))
        CPUBrandString.erase(i, n);

    return CPUBrandString;
}

string getLinuxOs()
{
    struct utsname unameData;
    uname(&unameData);

    string bit = unameData.machine;

    if (bit == "x86_64")
        bit = "64 bit";
    else if (bit == "i686")
        bit = "32 bit";

    return string(unameData.sysname) + " " + bit;
}
#elif defined(APPLE)
string systemInfo::getAppleCpuBrand()
{
    // https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man3/sysctl.3.html#//apple_ref/doc/man/3/sysctl
    //http://stackoverflow.com/questions/853798/programmatically-get-processor-details-from-mac-os-x
    //http://osxdaily.com/2011/07/15/get-cpu-info-via-command-line-in-mac-os-x/
#include <sys/types.h>
#include <sys/sysctl.h>

    string CPU = system("sysctl - n machdep.cpu.brand_string");

    CPU = trim(CPU);
    return CPU;
}
#endif

string systemInfo::getOs()
{
    string Os;
#if defined(WINDOWS)
    Os = GetWindowsVersion();
#elif defined(UNIX) || defined(LINUX)
    Os = getLinuxOs();
#elif defined(APPLE)
    Os = "Mac OSX";
#else
    Os = "OsUnkwonwn";
#endif
    return Os;
}

pair<string, string> systemInfo::getCompiler()
{
    pair<string, string> compiler;

#if defined(__clang__)
    /* Clang/LLVM. ---------------------------------------------- */
    compiler.first = "Clang";
    compiler.second = string(__clang_major__) + "_" + string(__clang_minor__);

#elif defined(__ICC) || defined(__INTEL_COMPILER)
    /* Intel ICC/ICPC. ------------------------------------------ */
    compiler.first = "IntelC++";
    compiler.second = string(__INTEL_COMPILER);

#elif defined(__GNUC__) || defined(__GNUG__)
    /* GNU GCC/G++. --------------------------------------------- */
    compiler.first = "GCC-G++";

#ifdef __VERSION__

    compiler.second = "_" + string(__VERSION__);
    replace(compiler.second.begin(), compiler.second.end(), ' ', '_');

#endif // __VERSION__

#elif defined(__HP_cc) || defined(__HP_aCC)
    /* Hewlett-Packard C/C++. ---------------------------------- */
    compiler.first = "HP_C++";

#elif defined(__IBMC__) || defined(__IBMCPP__)
    /* IBM XL C/C++. -------------------------------------------- */
    compiler.first = "IBM_XL_C-C++";

#elif defined(_MSC_VER)
    /* Microsoft Visual Studio. --------------------------------- */
    compiler.first = "MSVC";
    if (_MSC_VER == 1910) //Visual Studio 2017, MSVC++ 15.0
        compiler.second = "15.0";
    else if (_MSC_VER == 1900)
        compiler.second = "14.0";
    else if (_MSC_VER == 1800)
        compiler.second = "12.0";
    else if (_MSC_VER == 1700)
        compiler.second = "11.0";
    else if (_MSC_VER == 1600)
        compiler.second = "10.0";
    else if (_MSC_VER == 1500)
        compiler.second = "9.0";
    else if (_MSC_VER == 1400)
        compiler.second = "8.0";
    else if (_MSC_VER == 1310)
        compiler.second = "7.1";
    else if (_MSC_VER == 1300)
        compiler.second = "7.0";
    else if (_MSC_VER == 1200)
        compiler.second = "6.0";

    // else "None"

#elif defined(__PGI)
    /* Portland Group PGCC/PGCPP. ------------------------------- */
    compiler.first = "PGCC-PGCPP";

#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
    /* Oracle Solaris Studio. ----------------------------------- */
    compiler.first = "SUNPRO";

#endif
    return compiler;
}

ostream& operator<<(ostream& out, const systemInfo& sInfo)
{
    string str = sInfo.build + "\\_" + sInfo.compiler.first + sInfo.compiler.second + "\\_" + sInfo.os;
    string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
    str.erase(end_pos, str.end());
    out << str;
    return out;
}