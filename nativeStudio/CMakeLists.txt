cmake_minimum_required(VERSION 3.28)

set(GEM16_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
file(STRINGS "${GEM16_ROOT}/VERSION" GEM16_VERSION LIMIT_COUNT 1)
string(STRIP "${GEM16_VERSION}" GEM16_VERSION)
if(NOT GEM16_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(-[0-9A-Za-z.-]+)?(\\+[0-9A-Za-z.-]+)?$")
  message(FATAL_ERROR "VERSION must contain one semantic version, got '${GEM16_VERSION}'")
endif()
set(GEM16_CMAKE_VERSION
    "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")

project(gem16-studio VERSION "${GEM16_CMAKE_VERSION}" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")

set(IMGUI_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/imgui")

find_package(Threads REQUIRED)
include(cmake/markdown-dependencies.cmake)

set(STUDIO_SOURCES
  src/app.cpp
  src/audio_recorder.cpp
  src/api_client.cpp
  src/chat_history.cpp
  src/fonts.cpp
  src/image_texture.cpp
  src/main.cpp
  src/markdown.cpp
  src/markdown_parser.cpp
  src/math_renderer.cpp
  src/media_loader.cpp
  src/model_catalog.cpp
  src/model_manager.cpp
  src/platform_process.cpp
  src/platform_ui.cpp
  src/selectable_text.cpp
  src/server_manager.cpp
  src/settings.cpp
  src/shader_background.cpp
  "${GEM16_ROOT}/src/util/json.cpp"
  "${GEM16_ROOT}/src/compiler/sha256.cpp"
  "${GEM16_ROOT}/third_party/cpp-httplib/httplib.cpp"
  "${IMGUI_DIR}/imgui.cpp"
  "${IMGUI_DIR}/imgui_draw.cpp"
  "${IMGUI_DIR}/imgui_tables.cpp"
  "${IMGUI_DIR}/imgui_widgets.cpp"
)

if(WIN32)
  list(APPEND STUDIO_SOURCES
    "${IMGUI_DIR}/backends/imgui_impl_dx11.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_win32.cpp"
  )
elseif(UNIX AND NOT APPLE)
  include(FetchContent)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(glfw)
  find_package(OpenGL REQUIRED)
  list(APPEND STUDIO_SOURCES
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
  )
else()
  message(FATAL_ERROR "gem16 Studio currently supports Windows and Linux")
endif()

add_executable(gem16-studio ${STUDIO_SOURCES})
studio_markdown_dependencies(gem16-studio)
target_include_directories(gem16-studio PRIVATE
  src
  "${GEM16_ROOT}/include"
  "${GEM16_ROOT}/src"
  "${GEM16_ROOT}/third_party/cpp-httplib"
  "${GEM16_ROOT}/third_party/miniaudio"
  "${GEM16_ROOT}/third_party/stb"
  "${IMGUI_DIR}"
  "${IMGUI_DIR}/backends"
)
target_compile_definitions(gem16-studio PRIVATE
  IMGUI_DEFINE_MATH_OPERATORS
  IMGUI_USE_WCHAR32
  CPPHTTPLIB_TCP_NODELAY=1
  GEM16_REPOSITORY_ROOT="${GEM16_ROOT}"
  GEM16_VERSION_STRING="${GEM16_VERSION}"
)
target_link_libraries(gem16-studio PRIVATE Threads::Threads)

# GCC 16 emits a known std::variant false positive in the shared JSON parser.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_property(SOURCE "${GEM16_ROOT}/src/util/json.cpp" APPEND PROPERTY COMPILE_OPTIONS
    -Wno-maybe-uninitialized)
endif()

if(MSVC)
  set_target_properties(gem16-studio PROPERTIES
    WIN32_EXECUTABLE TRUE
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  )
  target_compile_definitions(gem16-studio PRIVATE
    UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_compile_options(gem16-studio PRIVATE /W4 /permissive- /utf-8)
  target_link_libraries(gem16-studio PRIVATE d3d11 dxgi d3dcompiler dwmapi ws2_32 comdlg32 shell32 ole32)
elseif(UNIX AND NOT APPLE)
  target_compile_options(gem16-studio PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(gem16-studio PRIVATE glfw OpenGL::GL dl PkgConfig::GTK3)
endif()

install(TARGETS gem16-studio RUNTIME DESTINATION bin)
install(FILES "${GEM16_ROOT}/docs/gem16_logo.svg"
        DESTINATION share/gem16 RENAME gem16-studio.svg)

include(CTest)
if(BUILD_TESTING)
  add_executable(gem16-studio-tests
    tests/studio_tests.cpp
    tests/markdown_render_tests.cpp
    src/api_client.cpp
    src/chat_history.cpp
    src/fonts.cpp
    src/image_texture.cpp
    src/markdown.cpp
    src/markdown_parser.cpp
    src/math_renderer.cpp
    src/media_loader.cpp
    src/model_catalog.cpp
    src/model_manager.cpp
    src/platform_process.cpp
    src/platform_ui.cpp
    src/selectable_text.cpp
    src/server_manager.cpp
    src/settings.cpp
    "${GEM16_ROOT}/src/util/json.cpp"
    "${GEM16_ROOT}/src/compiler/sha256.cpp"
    "${GEM16_ROOT}/third_party/cpp-httplib/httplib.cpp"
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
  )
  target_include_directories(gem16-studio-tests PRIVATE
    src
    "${GEM16_ROOT}/include"
    "${GEM16_ROOT}/src"
    "${GEM16_ROOT}/third_party/cpp-httplib"
    "${GEM16_ROOT}/third_party/stb"
    "${IMGUI_DIR}"
  )
  studio_markdown_dependencies(gem16-studio-tests)
  target_compile_definitions(gem16-studio-tests PRIVATE
    IMGUI_USE_WCHAR32
    CPPHTTPLIB_TCP_NODELAY=1
    GEM16_REPOSITORY_ROOT="${GEM16_ROOT}"
  )
  target_link_libraries(gem16-studio-tests PRIVATE Threads::Threads)
  if(WIN32)
    target_compile_options(gem16-studio-tests PRIVATE /utf-8)
    target_link_libraries(gem16-studio-tests PRIVATE d3d11 ws2_32 comdlg32 shell32 ole32)
    target_compile_definitions(gem16-studio-tests PRIVATE
      UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
    set_target_properties(gem16-studio-tests PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
  elseif(UNIX AND NOT APPLE)
    target_link_libraries(gem16-studio-tests PRIVATE OpenGL::GL PkgConfig::GTK3 dl)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(SOURCE "${GEM16_ROOT}/src/util/json.cpp" APPEND PROPERTY COMPILE_OPTIONS
      -Wno-maybe-uninitialized)
  endif()
  add_test(NAME gem16-native-studio-host COMMAND gem16-studio-tests)
endif()
