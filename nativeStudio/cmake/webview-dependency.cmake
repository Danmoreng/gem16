# Only a system WebView is used. Never fetch or distribute a browser engine.
target_sources(gem16-studio PRIVATE src/canvas_browser.cpp)
if(WIN32)
  include(FetchContent)
  FetchContent_Declare(studio_webview2
    URL https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/1.0.3719.77/microsoft.web.webview2.1.0.3719.77.nupkg
    URL_HASH SHA256=2f6be3a10a1a8d6d1fde986af4131dab344f8181fffe75590824e0f4b037ed73
    DOWNLOAD_NAME webview2-sdk.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_GetProperties(studio_webview2)
  if(NOT studio_webview2_POPULATED)
    FetchContent_Populate(studio_webview2)
  endif()
  target_compile_definitions(gem16-studio PRIVATE GEM16_WITH_WEBVIEW2=1)
  target_include_directories(gem16-studio PRIVATE "${studio_webview2_SOURCE_DIR}/build/native/include")
  # The small static loader locates the installed Evergreen runtime. No fixed runtime.
  target_link_libraries(gem16-studio PRIVATE
    "${studio_webview2_SOURCE_DIR}/build/native/x64/WebView2LoaderStatic.lib" version shlwapi dcomp)
  add_custom_command(TARGET gem16-studio POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:gem16-studio>/licenses"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${studio_webview2_SOURCE_DIR}/LICENSE.txt" "$<TARGET_FILE_DIR:gem16-studio>/licenses/WebView2-LICENSE.txt"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${studio_webview2_SOURCE_DIR}/NOTICE.txt" "$<TARGET_FILE_DIR:gem16-studio>/licenses/WebView2-NOTICE.txt")
else()
  pkg_check_modules(WEBKIT REQUIRED IMPORTED_TARGET webkit2gtk-4.1>=2.40)
  target_compile_definitions(gem16-studio PRIVATE GEM16_WITH_WEBKIT=1)
  target_link_libraries(gem16-studio PRIVATE PkgConfig::WEBKIT)
endif()
