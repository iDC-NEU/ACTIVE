#include <fstream>
#include <string>
#include <unistd.h> // sysconf
#include <iostream>

static inline double KB_to_GB(long kb)
{
    return kb / 1024.0 / 1024.0;
}

// 读取 /proc/self/status 里的 VmRSS / VmSize（单位 kB）
static void GetProcMemKB(long &rss_kb, long &vmsize_kb)
{
    rss_kb = 0;
    vmsize_kb = 0;
    std::ifstream in("/proc/self/status");
    std::string key;
    while (in >> key)
    {
        if (key == "VmRSS:")
        {
            in >> rss_kb; // kB
        }
        else if (key == "VmSize:")
        {
            in >> vmsize_kb; // kB
        }
        // 跳过该行剩余内容（单位、换行）
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

static void PrintMemUsage(const char *tag, int i, int total)
{
    long rss_kb, vmsize_kb;
    GetProcMemKB(rss_kb, vmsize_kb);
    std::cout << tag
              << " i=" << i << "/" << total
              << " VmRSS=" << KB_to_GB(rss_kb) << " GB"
              << " VmSize=" << KB_to_GB(vmsize_kb) << " GB"
              << std::endl;
}
