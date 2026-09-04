include(FetchContent)
# Official 3.53.4 amalgamation; archive SHA3 verified against sqlite.org/download.html.
FetchContent_Declare(studio_sqlite_source
  URL https://sqlite.org/2026/sqlite-amalgamation-3530400.zip
  URL_HASH SHA256=1e71ddf93849c6a6ecf58b827c0692073d2dd7ee40196158068f7b29f422e87d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR unused)
FetchContent_MakeAvailable(studio_sqlite_source)
add_library(studio_sqlite STATIC "${studio_sqlite_source_SOURCE_DIR}/sqlite3.c")
target_include_directories(studio_sqlite SYSTEM PUBLIC "${studio_sqlite_source_SOURCE_DIR}")
target_compile_definitions(studio_sqlite PRIVATE SQLITE_ENABLE_FTS5 SQLITE_OMIT_LOAD_EXTENSION SQLITE_DQS=0)
target_link_libraries(studio_sqlite PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if(MSVC)
  set_target_properties(studio_sqlite PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
