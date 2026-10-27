// Force-included into the CS529 compat build. libc++ (macOS) has no std::chrono::utc_clock yet,
// which the Dragon Logging Library headers name in a declaration.
#pragma once
#include <chrono>
#if !defined(__cpp_lib_chrono) || __cpp_lib_chrono < 201907L
namespace std::chrono
{
using utc_clock = system_clock;
template <typename Duration> using utc_time = time_point<utc_clock, Duration>;
}
#endif
