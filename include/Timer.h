#include <chrono>
#include <iostream>

class Timer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    Timer() : running(false), elapsed_time(0.0) {}

    // 开始计时
    void start()
    {
        start_time = Clock::now();
        running = true;
    }

    // 停止计时，并更新累计用时
    void stop()
    {
        if (running)
        {
            auto end_time = Clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            elapsed_time += duration.count();
            running = false;
        }
    }

    // 重置计时器
    void reset()
    {
        running = false;
        elapsed_time = 0.0;
    }

    // 获取总耗时（秒）
    double elapsed_seconds() const
    {
        return elapsed_time;
    }

    // 获取总耗时（毫秒）
    double elapsed_milliseconds() const
    {
        return elapsed_time * 1000.0;
    }

private:
    Clock::time_point start_time;
    bool running;
    double elapsed_time; // 累积时间（单位：秒）
};
