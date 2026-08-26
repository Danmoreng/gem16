cmake_minimum_required(VERSION 3.28)

project(gem16-studio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")

set(GEM16_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(IMGUI_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/imgui")

find_package(Threads REQUIRED)

set(STUDIO_SOURCES
  src/app.cpp
  src/api_client.cpp
  src/main.cpp
  src/platform_process.cpp
  src/selectable_text.cpp
  src/server_manager.cpp
  src/settings.cpp
  src/shader_background.cpp
  "${GEM16_ROOT}/src/util/json.cpp"
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
target_include_directories(gem16-studio PRIVATE
  src
  "${GEM16_ROOT}/include"
  "${GEM16_ROOT}/src"
  "${GEM16_ROOT}/third_party/cpp-httplib"
  "${IMGUI_DIR}"
  "${IMGUI_DIR}/backends"
)
target_compile_definitions(gem16-studio PRIVATE
  IMGUI_DEFINE_MATH_OPERATORS
  CPPHTTPLIB_TCP_NODELAY=1
  GEM16_REPOSITORY_ROOT="${GEM16_ROOT}"
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
  target_link_libraries(gem16-studio PRIVATE d3d11 dxgi d3dcompiler dwmapi ws2_32)
elseif(UNIX AND NOT APPLE)
  target_compile_options(gem16-studio PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(gem16-studio PRIVATE glfw OpenGL::GL dl)
endif()

install(TARGETS gem16-studio RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
  add_executable(gem16-studio-tests
    tests/studio_tests.cpp
    src/api_client.cpp
    src/platform_process.cpp
    src/selectable_text.cpp
    src/server_manager.cpp
    src/settings.cpp
    "${GEM16_ROOT}/src/util/json.cpp"
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
    "${IMGUI_DIR}"
  )
  target_compile_definitions(gem16-studio-tests PRIVATE
    CPPHTTPLIB_TCP_NODELAY=1
    GEM16_REPOSITORY_ROOT="${GEM16_ROOT}"
  )
  target_link_libraries(gem16-studio-tests PRIVATE Threads::Threads)
  if(WIN32)
    target_link_libraries(gem16-studio-tests PRIVATE ws2_32)
    target_compile_definitions(gem16-studio-tests PRIVATE
      UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
    set_target_properties(gem16-studio-tests PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(SOURCE "${GEM16_ROOT}/src/util/json.cpp" APPEND PROPERTY COMPILE_OPTIONS
      -Wno-maybe-uninitialized)
  endif()
  add_test(NAME gem16-native-studio-host COMMAND gem16-studio-tests)
endif()
