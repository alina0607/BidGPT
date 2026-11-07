include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(sell_anything
  src/main.cpp
  src/MlClient.cpp
  src/CanvasScene.cpp
  src/ConfirmScene.cpp
  src/RoundScene.cpp
  src/BuyScene.cpp
  src/CompareScene.cpp
  src/Characters.cpp
  src/Stage.cpp
  src/Shelter.cpp
  src/DoodleBoard.cpp
  src/Girl.cpp
  src/Sheets.cpp
  src/Npc.cpp
  src/Crowd.cpp
)
# The art is read from the source tree unless SA_ASSETS points elsewhere (see Characters.cpp).
target_compile_definitions(sell_anything PRIVATE SA_SOURCE_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/assets"
                                          SA_GAME_VERSION="${PROJECT_VERSION}")

target_link_libraries(sell_anything PRIVATE sa::engine nlohmann_json::nlohmann_json)
# The doodle board's text is Chinese, written as UTF-8 string literals.
if(MSVC)
  target_compile_options(sell_anything PRIVATE /utf-8)
endif()
if(WIN32)
  target_link_libraries(sell_anything PRIVATE ws2_32)
endif()

# Copy DGL.dll / DGL_d.dll next to the executable. Guarded because copy_if_different
# fails when the DLL list is empty.
if(SA_BACKEND_DGL)
  add_custom_command(TARGET sell_anything POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:sell_anything> $<TARGET_FILE_DIR:sell_anything>
    COMMAND_EXPAND_LISTS
  )
endif()
