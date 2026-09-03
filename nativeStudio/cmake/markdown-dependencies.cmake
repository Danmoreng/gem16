include(FetchContent)
# Same upstream versions as Simple Markdown Viewer, pinned to immutable commits.
FetchContent_Declare(md4c GIT_REPOSITORY https://github.com/mity/md4c.git
  GIT_TAG 729e6b8b320caa96328968ab27d7db2235e4fb47 SOURCE_SUBDIR unused)
FetchContent_Declare(microtex GIT_REPOSITORY https://github.com/NanoMichael/MicroTeX.git
  GIT_TAG 0e3707f6dafebb121d98b53c64364d16fefe481d SOURCE_SUBDIR unused)
FetchContent_Declare(tinyxml2 GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
  GIT_TAG 1dee28e51f9175a31955b9791c74c430fe13dc82 SOURCE_SUBDIR unused)
FetchContent_MakeAvailable(md4c microtex tinyxml2)
# Reproducible derived source, retaining the upstream checkout unchanged.
# Upstream uses delete on an asprintf allocation in the Linux HOME path.
file(READ "${microtex_SOURCE_DIR}/src/latex.cpp" tex_latex)
string(REPLACE "    char* userdata_fallback;\n    asprintf(&userdata_fallback, \"%s/.local/share/clatexmath/\", home);\n    paths.push(string(userdata_fallback));\n    delete userdata_fallback;"
  "    paths.push(string(home) + \"/.local/share/clatexmath/\");" tex_latex "${tex_latex}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/microtex-latex.cpp" CONTENT "${tex_latex}" @ONLY)
add_library(studio_md4c STATIC "${md4c_SOURCE_DIR}/src/md4c.c" "${md4c_SOURCE_DIR}/src/entity.c")
target_include_directories(studio_md4c PUBLIC "${md4c_SOURCE_DIR}/src")
file(GLOB_RECURSE tex_sources CONFIGURE_DEPENDS
  "${microtex_SOURCE_DIR}/src/atom/*.cpp" "${microtex_SOURCE_DIR}/src/box/*.cpp"
  "${microtex_SOURCE_DIR}/src/core/*.cpp" "${microtex_SOURCE_DIR}/src/fonts/*.cpp"
  "${microtex_SOURCE_DIR}/src/res/*.cpp" "${microtex_SOURCE_DIR}/src/utils/*.cpp")
add_library(studio_microtex STATIC ${tex_sources}
  "${CMAKE_CURRENT_BINARY_DIR}/microtex-latex.cpp" "${microtex_SOURCE_DIR}/src/render.cpp"
  "${tinyxml2_SOURCE_DIR}/tinyxml2.cpp")
target_include_directories(studio_microtex SYSTEM PUBLIC "${microtex_SOURCE_DIR}/src"
  PRIVATE "${tinyxml2_SOURCE_DIR}")
if(MSVC)
  set_target_properties(studio_md4c studio_microtex PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
function(studio_markdown_dependencies target)
  target_link_libraries(${target} PRIVATE studio_md4c studio_microtex)
  target_compile_definitions(${target} PRIVATE GEM16_MATH_RESOURCE_DIR="${microtex_SOURCE_DIR}/res")
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${microtex_SOURCE_DIR}/res" "$<TARGET_FILE_DIR:${target}>/math-res"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/licenses"
    COMMAND ${CMAKE_COMMAND} -E copy "${md4c_SOURCE_DIR}/LICENSE.md" "$<TARGET_FILE_DIR:${target}>/licenses/md4c-MIT.txt"
    COMMAND ${CMAKE_COMMAND} -E copy "${microtex_SOURCE_DIR}/LICENSE" "$<TARGET_FILE_DIR:${target}>/licenses/MicroTeX-MIT.txt"
    COMMAND ${CMAKE_COMMAND} -E copy "${tinyxml2_SOURCE_DIR}/LICENSE.txt" "$<TARGET_FILE_DIR:${target}>/licenses/tinyxml2-zlib.txt")
endfunction()
install(DIRECTORY "${microtex_SOURCE_DIR}/res/" DESTINATION bin/math-res)
install(FILES "${md4c_SOURCE_DIR}/LICENSE.md" DESTINATION licenses RENAME md4c-MIT.txt)
install(FILES "${microtex_SOURCE_DIR}/LICENSE" DESTINATION licenses RENAME MicroTeX-MIT.txt)
install(FILES "${tinyxml2_SOURCE_DIR}/LICENSE.txt" DESTINATION licenses RENAME tinyxml2-zlib.txt)
