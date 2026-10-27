// See LoggingSystem.h.

#include "Precompiled.h"

#include "LoggingSystem.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace dragon
{
	// Declared by <dragon/assert.h>; normally defined in dragon.lib.
	bool assert_enabled = true;
}

namespace CS529
{
	LoggingSystem* LoggingSystem::instance = nullptr;

	LoggingSystem::LoggingSystem(std::string_view applicationName, std::filesystem::path const& path)
		: System("LoggingSystem")
		, applicationName(applicationName)
		, path(path)
	{
		assert(instance == nullptr);
		instance = this;
	}

	LoggingSystem::~LoggingSystem(void)
	{
		instance = nullptr;
	}

	bool LoggingSystem::Initialize()
	{
		file.open(path, std::ios::out | std::ios::trunc);
		if (!file)
		{
			std::fprintf(stderr, "LoggingSystem: could not open %s\n", path.string().c_str());
		}
		Emit("info", std::format("{} started", applicationName));
		return true;
	}

	void LoggingSystem::Emit(std::string_view level, std::string_view message) noexcept
	{
		// Set CS529_LOG_QUIET=1 to keep stderr clean; the trace file still gets everything.
		static const bool quiet = std::getenv("CS529_LOG_QUIET") != nullptr;

		if (!instance)
		{
			std::fprintf(stderr, "[%.*s] %.*s\n", static_cast<int>(level.size()), level.data(),
			             static_cast<int>(message.size()), message.data());
			return;
		}

		std::lock_guard lock(instance->mutex);
		if (instance->file)
		{
			instance->file << '[' << level << "] " << message << '\n';
		}
		if (!quiet && level != "debug" && level != "verbose")
		{
			std::fprintf(stderr, "[%.*s] %.*s\n", static_cast<int>(level.size()), level.data(),
			             static_cast<int>(message.size()), message.data());
		}
	}
}	// namespace
