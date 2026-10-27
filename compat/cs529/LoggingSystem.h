// Drop-in replacement for the CS529 LoggingSystem, used only by the macOS/Linux compat build.
// The original wraps the Dragon Logging Library, which ships as a Windows-only dragon.lib.
// Same public interface; writes to the trace file and to stderr.

#pragma once

#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string_view>

#include "System.h"

namespace CS529
{
	class LoggingSystem : public System
	{
	public:
		LoggingSystem(std::string_view applicationName, std::filesystem::path const& path);
		~LoggingSystem(void) override;

		template <typename... Args>
		static void Debug(std::format_string<Args...> const format, Args&&... args) noexcept
		{
			Write("debug", format, std::forward<Args>(args)...);
		}

		template <typename... Args>
		static void Verbose(std::format_string<Args...> const format, Args&&... args) noexcept
		{
			Write("verbose", format, std::forward<Args>(args)...);
		}

		template <typename... Args>
		static void Info(std::format_string<Args...> const format, Args&&... args) noexcept
		{
			Write("info", format, std::forward<Args>(args)...);
		}

		template <typename... Args>
		static void Warning(std::format_string<Args...> const format, Args&&... args) noexcept
		{
			Write("warning", format, std::forward<Args>(args)...);
		}

		template <typename... Args>
		static void Error(std::format_string<Args...> const format, Args&&... args) noexcept
		{
			Write("error", format, std::forward<Args>(args)...);
		}

	private:
		bool Initialize() override;

		template <typename... Args>
		static void Write(std::string_view level, std::format_string<Args...> const format,
		                  Args&&... args) noexcept
		{
			try
			{
				Emit(level, std::format(format, std::forward<Args>(args)...));
			}
			catch (...)
			{
			}
		}

		static void Emit(std::string_view level, std::string_view message) noexcept;

		static LoggingSystem* instance;

		std::string_view applicationName;
		std::filesystem::path path;
		std::ofstream file;
		std::mutex mutex;
	};
}	// namespace
