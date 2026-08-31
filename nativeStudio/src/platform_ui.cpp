#include "platform_ui.h"

#include <mutex>
#include <optional>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>
#else
#include <array>
#include <gtk/gtk.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace gem16::studio {
namespace {

std::mutex dropped_mutex;
std::vector<std::filesystem::path> dropped_files;

#ifndef _WIN32
std::optional<std::vector<std::filesystem::path>> RunGtkPicker(
    const char* title, GtkFileChooserAction action, bool multiple) {
  if (!gtk_init_check(nullptr, nullptr)) return std::nullopt;
  GtkFileChooserNative* picker = gtk_file_chooser_native_new(
      title, nullptr, action, action == GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER
                                  ? "Select" : "Open", "Cancel");
  gtk_file_chooser_set_select_multiple(GTK_FILE_CHOOSER(picker), multiple);
  std::vector<std::filesystem::path> paths;
  if (gtk_native_dialog_run(GTK_NATIVE_DIALOG(picker)) == GTK_RESPONSE_ACCEPT) {
    GSList* files = gtk_file_chooser_get_filenames(GTK_FILE_CHOOSER(picker));
    for (GSList* node = files; node != nullptr; node = node->next) {
      const char* file = static_cast<const char*>(node->data);
      if (file && *file) paths.emplace_back(file);
      g_free(node->data);
    }
    g_slist_free(files);
  }
  g_object_unref(picker);
  while (gtk_events_pending()) gtk_main_iteration_do(false);
  return paths;
}

std::vector<std::filesystem::path> RunPicker(const char* executable,
                                             std::vector<std::string> arguments) {
  int output[2];
  if (pipe(output) != 0) return {};
  const pid_t child = fork();
  if (child < 0) {
    close(output[0]);
    close(output[1]);
    return {};
  }
  if (child == 0) {
    close(output[0]);
    dup2(output[1], STDOUT_FILENO);
    close(output[1]);
    std::vector<char*> argv{const_cast<char*>(executable)};
    for (std::string& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    execvp(executable, argv.data());
    _exit(127);
  }
  close(output[1]);
  std::string result;
  std::array<char, 4096> buffer{};
  for (ssize_t count = read(output[0], buffer.data(), buffer.size()); count > 0;
       count = read(output[0], buffer.data(), buffer.size())) {
    result.append(buffer.data(), static_cast<std::size_t>(count));
  }
  close(output[0]);
  int status = 0;
  waitpid(child, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return {};
  std::vector<std::filesystem::path> paths;
  std::size_t begin = 0;
  while (begin < result.size()) {
    const std::size_t end = result.find_first_of("|\n", begin);
    std::string path = result.substr(begin, end - begin);
    if (!path.empty() && path.back() == '\r') path.pop_back();
    if (!path.empty()) paths.emplace_back(std::move(path));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return paths;
}
#endif

}  // namespace

std::vector<std::filesystem::path> OpenAttachmentDialog() {
#ifdef _WIN32
  std::wstring buffer(65536, L'\0');
  const wchar_t filters[] =
      L"Supported files\0*.txt;*.md;*.csv;*.json;*.cpp;*.h;*.py;*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.wav;*.flac;*.mp3\0All files\0*.*\0\0";
  OPENFILENAMEW dialog{};
  dialog.lStructSize = sizeof(dialog);
  dialog.lpstrFilter = filters;
  dialog.lpstrFile = buffer.data();
  dialog.nMaxFile = static_cast<DWORD>(buffer.size());
  dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_ALLOWMULTISELECT |
                 OFN_EXPLORER | OFN_NOCHANGEDIR;
  if (!GetOpenFileNameW(&dialog)) return {};
  std::vector<std::filesystem::path> result;
  const std::filesystem::path first(buffer.c_str());
  const wchar_t* cursor = buffer.c_str() + first.native().size() + 1;
  if (*cursor == L'\0') return {first};
  while (*cursor != L'\0') {
    std::filesystem::path name(cursor);
    result.push_back(first / name);
    cursor += name.native().size() + 1;
  }
  return result;
#else
  if (auto paths = RunGtkPicker("Attach files to Gem 16", GTK_FILE_CHOOSER_ACTION_OPEN, true))
    return *paths;
  auto paths = RunPicker("zenity", {"--file-selection", "--multiple", "--separator=|",
                                "--title=Attach files to Gem 16"});
  if (!paths.empty()) return paths;
  return RunPicker("kdialog", {"--getopenfilename", ".", "*", "--multiple", "--separate-output"});
#endif
}

std::filesystem::path OpenExecutableDialog() {
#ifdef _WIN32
  std::wstring buffer(32768, L'\0');
  const wchar_t filters[] = L"Executable files\0*.exe\0All files\0*.*\0\0";
  OPENFILENAMEW dialog{};
  dialog.lStructSize = sizeof(dialog);
  dialog.lpstrFilter = filters;
  dialog.lpstrFile = buffer.data();
  dialog.nMaxFile = static_cast<DWORD>(buffer.size());
  dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  return GetOpenFileNameW(&dialog) ? std::filesystem::path(buffer.c_str())
                                   : std::filesystem::path{};
#else
  if (auto paths = RunGtkPicker("Select gem16-server", GTK_FILE_CHOOSER_ACTION_OPEN, false))
    return paths->empty() ? std::filesystem::path{} : paths->front();
  auto paths = RunPicker("zenity", {"--file-selection", "--title=Select gem16-server"});
  if (paths.empty()) paths = RunPicker("kdialog", {"--getopenfilename", "."});
  return paths.empty() ? std::filesystem::path{} : paths.front();
#endif
}

std::filesystem::path OpenDirectoryDialog() {
#ifdef _WIN32
  BROWSEINFOW info{};
  info.lpszTitle = L"Select model directory";
  info.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
  PIDLIST_ABSOLUTE item = SHBrowseForFolderW(&info);
  if (!item) return {};
  std::wstring buffer(32768, L'\0');
  const bool resolved = SHGetPathFromIDListW(item, buffer.data());
  CoTaskMemFree(item);
  return resolved ? std::filesystem::path(buffer.c_str()) : std::filesystem::path{};
#else
  if (auto paths = RunGtkPicker("Select model directory", GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, false))
    return paths->empty() ? std::filesystem::path{} : paths->front();
  auto paths = RunPicker("zenity", {"--file-selection", "--directory", "--title=Select model directory"});
  if (paths.empty()) paths = RunPicker("kdialog", {"--getexistingdirectory", "."});
  return paths.empty() ? std::filesystem::path{} : paths.front();
#endif
}

std::vector<std::filesystem::path> DrainDroppedFiles() {
  std::lock_guard lock(dropped_mutex);
  std::vector<std::filesystem::path> result;
  result.swap(dropped_files);
  return result;
}

void QueueDroppedFiles(const std::vector<std::filesystem::path>& paths) {
  std::lock_guard lock(dropped_mutex);
  dropped_files.insert(dropped_files.end(), paths.begin(), paths.end());
}

void OpenInFileManager(const std::filesystem::path& path) {
#ifdef _WIN32
  ShellExecuteW(nullptr, L"open", path.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
  const pid_t child = fork();
  if (child == 0) {
    execlp("xdg-open", "xdg-open", path.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
#endif
}

}  // namespace gem16::studio
