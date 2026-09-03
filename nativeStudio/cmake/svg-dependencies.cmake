include(FetchContent)
set(LUNASVG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(lunasvg
  GIT_REPOSITORY https://github.com/sammycage/lunasvg.git
  GIT_TAG 83c58df8103dc7dca423dfd824992af94d49bed6)
FetchContent_MakeAvailable(lunasvg)
# Retain the licenses embedded in PlutoVG's bundled stb headers as standalone
# generated notices; do not modify the pinned upstream sources.
foreach(component image image-write truetype)
  file(READ "${lunasvg_SOURCE_DIR}/plutovg/source/plutovg-stb-${component}.h" stb_source)
  string(FIND "${stb_source}" "ALTERNATIVE A - MIT License" notice_offset)
  if(notice_offset LESS 0)
    message(FATAL_ERROR "Missing PlutoVG stb ${component} license")
  endif()
  string(SUBSTRING "${stb_source}" ${notice_offset} -1 stb_notice)
  file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/PlutoVG-stb-${component}-notice.txt" CONTENT "${stb_notice}" @ONLY)
endforeach()
if(MSVC)
  set_target_properties(lunasvg plutovg PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
function(studio_svg_dependencies target)
  target_link_libraries(${target} PRIVATE lunasvg::lunasvg)
  target_include_directories(${target} SYSTEM PRIVATE "${tinyxml2_SOURCE_DIR}")
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/licenses"
    COMMAND ${CMAKE_COMMAND} -E copy "${lunasvg_SOURCE_DIR}/LICENSE" "$<TARGET_FILE_DIR:${target}>/licenses/LunaSVG-MIT.txt"
    COMMAND ${CMAKE_COMMAND} -E copy "${lunasvg_SOURCE_DIR}/plutovg/LICENSE" "$<TARGET_FILE_DIR:${target}>/licenses/PlutoVG-MIT.txt"
    COMMAND ${CMAKE_COMMAND} -E copy "${lunasvg_SOURCE_DIR}/plutovg/source/FTL.TXT" "$<TARGET_FILE_DIR:${target}>/licenses/PlutoVG-FreeType-FTL.txt")
  foreach(component image image-write truetype)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/PlutoVG-stb-${component}-notice.txt" "$<TARGET_FILE_DIR:${target}>/licenses/")
  endforeach()
endfunction()
install(FILES "${lunasvg_SOURCE_DIR}/LICENSE" DESTINATION licenses RENAME LunaSVG-MIT.txt)
install(FILES "${lunasvg_SOURCE_DIR}/plutovg/LICENSE" DESTINATION licenses RENAME PlutoVG-MIT.txt)
install(FILES "${lunasvg_SOURCE_DIR}/plutovg/source/FTL.TXT" DESTINATION licenses RENAME PlutoVG-FreeType-FTL.txt)
foreach(component image image-write truetype)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PlutoVG-stb-${component}-notice.txt" DESTINATION licenses)
endforeach()
