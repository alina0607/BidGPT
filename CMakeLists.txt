cmake_minimum_required(VERSION 3.24)
project(SellAnything VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Backends ----------------------------------------------------------------
# Every backend that is enabled gets compiled in; pick one at run time with --backend.
# DGL is not redistributable: point at it with -DSA_DGL_ROOT=..., the DGL_ROOT environment
# variable (set it once and every clone finds it), or by copying it into third_party/DGL.
if(DEFINED ENV{DGL_ROOT})
  set(_sa_dgl_default "$ENV{DGL_ROOT}")
else()
  set(_sa_dgl_default "${CMAKE_SOURCE_DIR}/third_party/DGL")
endif()
set(SA_DGL_ROOT "${_sa_dgl_default}" CACHE PATH "Folder containing DGL's inc/ and lib/x64/")

set(_sa_dgl_found OFF)
if(WIN32 AND EXISTS "${SA_DGL_ROOT}/inc/DGL.h")
  set(_sa_dgl_found ON)
endif()

option(SA_BACKEND_OPENGL "Build the GLFW + OpenGL backend" ON)
option(SA_BACKEND_DGL "Build the DGL (Direct3D 11) backend" ${_sa_dgl_found})

# DGL API on OpenGL, for running DGL programs (e.g. CS529 projects) off Windows. See compat/.
set(_sa_compat_default ON)
if(WIN32)
  set(_sa_compat_default OFF) # Windows has the real DGL
endif()
option(SA_BUILD_DGL_COMPAT "Build the DGL-on-OpenGL compatibility layer" ${_sa_compat_default})
# The DGL backend's own source, built against the compat layer: checks it off Windows.
option(SA_BACKEND_DGL_COMPAT "Build the DGL backend on the compat layer (macOS/Linux)" OFF)

if(SA_BACKEND_DGL AND NOT WIN32)
  message(FATAL_ERROR "SA_BACKEND_DGL requires Windows")
endif()
if(SA_BACKEND_DGL AND NOT EXISTS "${SA_DGL_ROOT}/inc/DGL.h")
  message(FATAL_ERROR "DGL not found at SA_DGL_ROOT=${SA_DGL_ROOT} (see docs/backends.md)")
endif()
if(SA_BACKEND_DGL_COMPAT AND (WIN32 OR NOT SA_BUILD_DGL_COMPAT))
  message(FATAL_ERROR "SA_BACKEND_DGL_COMPAT is for macOS/Linux and needs SA_BUILD_DGL_COMPAT")
endif()
if(SA_BUILD_DGL_COMPAT AND NOT SA_BACKEND_OPENGL)
  message(FATAL_ERROR "SA_BUILD_DGL_COMPAT needs SA_BACKEND_OPENGL (it shares GLFW and glad)")
endif()
if(WIN32 AND NOT SA_BACKEND_DGL)
  message(STATUS "DGL not found at ${SA_DGL_ROOT}; building without the DGL backend")
endif()

message(STATUS "Sell Anything backends: OpenGL=${SA_BACKEND_OPENGL} DGL=${SA_BACKEND_DGL} DGL-on-compat=${SA_BACKEND_DGL_COMPAT} null=ON")

# --- Warnings ----------------------------------------------------------------
if(MSVC)
  set(SA_WARNINGS /W4 /permissive-)
else()
  set(SA_WARNINGS -Wall -Wextra -Wpedantic)
endif()

add_subdirectory(engine)
add_subdirectory(game)
if(SA_BUILD_DGL_COMPAT)
  add_subdirectory(compat)
endif()

target_compile_options(sa_engine PRIVATE ${SA_WARNINGS})
target_compile_options(sell_anything PRIVATE ${SA_WARNINGS})

enable_testing()
add_test(NAME smoke_null_backend COMMAND sell_anything --backend null)

add_executable(text_test engine/tests/text_test.cpp)
target_link_libraries(text_test PRIVATE sa::engine)
target_compile_options(text_test PRIVATE ${SA_WARNINGS})
add_test(NAME text_layout COMMAND text_test)

add_executable(sprite_test engine/tests/sprite_test.cpp)
target_link_libraries(sprite_test PRIVATE sa::engine)
target_compile_options(sprite_test PRIVATE ${SA_WARNINGS})
add_test(NAME sprites COMMAND sprite_test "${CMAKE_SOURCE_DIR}/engine/tests/fixtures")

add_executable(mlclient_test game/tests/mlclient_test.cpp game/src/MlClient.cpp)
target_include_directories(mlclient_test PRIVATE game/src)
target_link_libraries(mlclient_test PRIVATE nlohmann_json::nlohmann_json)
if(WIN32)
  target_link_libraries(mlclient_test PRIVATE ws2_32)
endif()
target_compile_options(mlclient_test PRIVATE ${SA_WARNINGS})
add_test(NAME mlclient_timeout COMMAND mlclient_test)

add_executable(characters_test game/tests/characters_test.cpp game/src/Characters.cpp)
target_include_directories(characters_test PRIVATE game/src)
target_link_libraries(characters_test PRIVATE sa::engine nlohmann_json::nlohmann_json)
target_compile_definitions(characters_test PRIVATE SA_SOURCE_ASSETS="${CMAKE_SOURCE_DIR}/game/assets")
target_compile_options(characters_test PRIVATE ${SA_WARNINGS})
add_test(NAME characters COMMAND characters_test)

add_executable(girl_test game/tests/girl_test.cpp game/src/Girl.cpp game/src/Stage.cpp game/src/Sheets.cpp
  game/src/Npc.cpp game/src/Characters.cpp)
target_include_directories(girl_test PRIVATE game/src)
target_link_libraries(girl_test PRIVATE sa::engine nlohmann_json::nlohmann_json)
target_compile_definitions(girl_test PRIVATE SA_SOURCE_ASSETS="${CMAKE_SOURCE_DIR}/game/assets")
target_compile_options(girl_test PRIVATE ${SA_WARNINGS})
add_test(NAME girl COMMAND girl_test)
