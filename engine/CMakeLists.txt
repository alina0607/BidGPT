include(FetchContent)
# 8×8 ASCII bitmap font, public domain.
FetchContent_Declare(font8x8
  GIT_REPOSITORY https://github.com/dhepper/font8x8.git
  GIT_TAG 8e279d2d864e79128e96188a6b9526cfa3fbfef9
)
FetchContent_MakeAvailable(font8x8)
# stb_image (PNG loading for sprites) and stb_image_write (screenshots), public domain.
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
)
FetchContent_MakeAvailable(stb)

add_library(sa_engine
  src/Engine.cpp
  src/BackendRegistry.cpp
  src/Text.cpp
  src/Image.cpp
  src/Sprite.cpp
  src/font8x8_data.c
)
add_library(sa::engine ALIAS sa_engine)

target_include_directories(sa_engine
  PUBLIC include
  PRIVATE src
)
target_include_directories(sa_engine SYSTEM PRIVATE "${font8x8_SOURCE_DIR}" "${stb_SOURCE_DIR}")

# --- OpenGL backend (GLFW) ---------------------------------------------------
if(SA_BACKEND_OPENGL)
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(glfw)

  target_sources(sa_engine PRIVATE src/backends/OpenGLBackend.cpp)
  # GLFW loads the system OpenGL library at run time; glad (bundled with GLFW) loads the functions.
  target_include_directories(sa_engine SYSTEM PRIVATE "${glfw_SOURCE_DIR}/deps")   # glad, stb_image_write
  target_link_libraries(sa_engine PRIVATE glfw)
  target_compile_definitions(sa_engine PRIVATE SA_HAS_OPENGL=1)
endif()

# --- DGL backend (Windows, local SDK only) -----------------------------------
if(SA_BACKEND_DGL)
  add_library(DGL::DGL SHARED IMPORTED GLOBAL)
  set_target_properties(DGL::DGL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SA_DGL_ROOT}/inc"
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_IMPLIB_DEBUG "${SA_DGL_ROOT}/lib/x64/DGL_d.lib"
    IMPORTED_LOCATION_DEBUG "${SA_DGL_ROOT}/lib/x64/DGL_d.dll"
    IMPORTED_IMPLIB_RELEASE "${SA_DGL_ROOT}/lib/x64/DGL.lib"
    IMPORTED_LOCATION_RELEASE "${SA_DGL_ROOT}/lib/x64/DGL.dll"
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE
    MAP_IMPORTED_CONFIG_MINSIZEREL RELEASE
  )

  target_sources(sa_engine PRIVATE src/backends/DGLBackend.cpp)
  target_link_libraries(sa_engine PUBLIC DGL::DGL)
  target_compile_definitions(sa_engine PRIVATE SA_HAS_DGL=1 NOMINMAX)
endif()

# --- DGL backend on the compat layer (macOS/Linux) ----------------------------
# Same DGLBackend.cpp, linked against compat/ instead of DigiPen's DLL, so the code that ships for
# Windows is compiled and run on every machine (see docs/backends.md).
if(SA_BACKEND_DGL_COMPAT)
  target_sources(sa_engine PRIVATE src/backends/DGLBackend.cpp)
  target_link_libraries(sa_engine PUBLIC dgl_compat)
  # compat/ already compiles glad and stb into its library; one copy of each, or the link fails.
  target_compile_definitions(sa_engine PRIVATE SA_HAS_DGL=1 SA_THIRD_PARTY_FROM_COMPAT=1)
endif()
