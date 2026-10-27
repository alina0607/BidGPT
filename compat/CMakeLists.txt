# DGL on OpenGL: lets programs written against the DigiPen Graphics Library build and run on
# macOS and Linux. See compat/README.md.

include(FetchContent)
FetchContent_GetProperties(glfw) # fetched by engine/; we need its bundled glad
FetchContent_GetProperties(stb)  # fetched by engine/ too

add_library(dgl_compat STATIC dgl/src/DGL_GL.cpp)
add_library(sa::dgl_compat ALIAS dgl_compat)
target_compile_features(dgl_compat PUBLIC cxx_std_20)
target_include_directories(dgl_compat
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/dgl/include/DGL/inc"   # #include "DGL.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/dgl/include"           # #include <DGL/inc/DGL.h>
    "${CMAKE_CURRENT_SOURCE_DIR}/win32/include"         # <Windows.h>, <sal.h>, ...
)
# Third-party headers: glad (bundled with GLFW) and stb_image / stb_image_write.
target_include_directories(dgl_compat SYSTEM PRIVATE "${stb_SOURCE_DIR}" "${glfw_SOURCE_DIR}/deps")
target_link_libraries(dgl_compat PUBLIC glfw)
target_compile_options(dgl_compat PRIVATE ${SA_WARNINGS})

# Linux file systems are case-sensitive; sources spell it both <Windows.h> and <windows.h>.
if(NOT APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/case/windows.h" "#pragma once\n#include <Windows.h>\n")
  target_include_directories(dgl_compat PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/case")
endif()

# Self-contained demo, so the layer is built and smoke-tested without any course code.
add_executable(dgl_demo examples/dgl_demo.cpp win32/src/WinMainBridge.cpp)
target_link_libraries(dgl_demo PRIVATE dgl_compat)
target_compile_options(dgl_demo PRIVATE ${SA_WARNINGS})

# --- Build a CS529 project against the compat layer ---------------------------
# cmake -S . -B build -DSA_CS529_PROJECT="/path/to/Project3-Transformations (Student)"
#
# The project's Source/ is copied into the build tree on every build (originals are never
# modified), with LoggingSystem swapped for a version that doesn't need dragon.lib.
set(SA_CS529_PROJECT "" CACHE PATH "CS529 project folder (the one containing Source/) to build")

if(SA_CS529_PROJECT)
  if(NOT EXISTS "${SA_CS529_PROJECT}/Source/Main.cpp")
    message(FATAL_ERROR "SA_CS529_PROJECT=${SA_CS529_PROJECT} has no Source/Main.cpp")
  endif()

  set(_src_dir "${SA_CS529_PROJECT}/Source")
  set(_out_dir "${CMAKE_CURRENT_BINARY_DIR}/cs529_src")
  set(_overlay "${CMAKE_CURRENT_SOURCE_DIR}/cs529")
  # LoggingSystem: replaced. SimpleTextFileChannel: only used by the original LoggingSystem.
  set(_replaced LoggingSystem.h LoggingSystem.cpp)
  set(_skipped SimpleTextFileChannel.cpp)

  file(GLOB _project_files CONFIGURE_DEPENDS "${_src_dir}/*.cpp" "${_src_dir}/*.h")
  set(_copied)
  foreach(_file IN LISTS _project_files)
    get_filename_component(_name "${_file}" NAME)
    if(_name IN_LIST _skipped)
      continue()
    endif()
    if(_name IN_LIST _replaced)
      set(_from "${_overlay}/${_name}")
    else()
      set(_from "${_file}")
    endif()
    add_custom_command(
      OUTPUT "${_out_dir}/${_name}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_from}" "${_out_dir}/${_name}"
      DEPENDS "${_from}"
      COMMENT "Syncing ${_name}"
      VERBATIM
    )
    list(APPEND _copied "${_out_dir}/${_name}")
  endforeach()

  add_executable(cs529_app ${_copied} win32/src/WinMainBridge.cpp)
  target_compile_features(cs529_app PRIVATE cxx_std_23)
  # Order matters: the compat DGL.h must win over the real one in <project>/Libraries/DGL.
  target_link_libraries(cs529_app PRIVATE dgl_compat)
  target_include_directories(cs529_app PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/dgl/include/DGL/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/dgl/include"
    "${_out_dir}"
    "${SA_CS529_PROJECT}/Libraries"
    "${SA_CS529_PROJECT}/Libraries/DLL/include"
  )
  target_compile_options(cs529_app PRIVATE
    -include "${CMAKE_CURRENT_SOURCE_DIR}/cs529/libcxx_compat.h"
    # Course code is written for MSVC; keep clang quiet about what MSVC accepts.
    -Wno-unknown-attributes -Wno-ignored-attributes -Wno-unknown-pragmas
    -Wno-nonportable-include-path -Wno-invalid-utf8
    -Wno-inconsistent-missing-override -Wno-potentially-evaluated-expression
  )
  # The CheatSystem (scene hot-keys) only exists in debug builds, same as Visual Studio.
  target_compile_definitions(cs529_app PRIVATE $<$<CONFIG:Debug>:_DEBUG> _WINDOWS)

  # Assets/ and Data/ are opened with relative paths. Run from a folder in the build tree that
  # links to them, so Trace.log and other output never land in the course project.
  set(SA_CS529_RUN_DIR "${CMAKE_CURRENT_BINARY_DIR}/cs529_run" CACHE INTERNAL "")
  file(MAKE_DIRECTORY "${SA_CS529_RUN_DIR}")
  foreach(_dir Assets Data)
    if(EXISTS "${SA_CS529_PROJECT}/${_dir}")
      file(CREATE_LINK "${SA_CS529_PROJECT}/${_dir}" "${SA_CS529_RUN_DIR}/${_dir}" SYMBOLIC)
    endif()
  endforeach()
  add_custom_target(run_cs529
    COMMAND cs529_app
    WORKING_DIRECTORY "${SA_CS529_RUN_DIR}"
    DEPENDS cs529_app
    USES_TERMINAL
  )
  message(STATUS "CS529 compat build: ${SA_CS529_PROJECT}")
endif()
