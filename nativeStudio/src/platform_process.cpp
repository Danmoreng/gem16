#include "platform_process.h"

#include <array>
#include <chrono>
#include <mutex>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace gem16::studio {

struct PlatformProcess::Impl {
  std::atomic<bool> running{false};
  std::mutex stop_mutex;
  std::jthread reader;
  std::jthread waiter;
#ifdef _WIN32
  HANDLE process = nullptr;
  HANDLE thread = nullptr;
  HANDLE output = nullptr;
#else
  pid_t process = -1;
  int output = -1;
#endif
};

PlatformProcess::PlatformProcess() : impl_(std::make_unique<Impl>()) {}

PlatformProcess::~PlatformProcess() { Stop(); }

#ifdef _WIN32
namespace {
std::wstring Utf8ToWide(const std::string& value) {
  if (value.empty()) return {};
  const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                         static_cast<int>(value.size()), nullptr, 0);
  if (length <= 0) return {};
  std::wstring result(static_cast<std::size_t>(length), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                      static_cast<int>(value.size()), result.data(), length);
  return result;
}

std::wstring QuoteArgument(const std::string& value) {
  const std::wstring source = Utf8ToWide(value);
  if (source.find_first_of(L" \t\"") == std::wstring::npos) return source;
  std::wstring result = L"\"";
  std::size_t backslashes = 0;
  for (wchar_t character : source) {
    if (character == L'\\') {
      ++backslashes;
    } else if (character == L'\"') {
      result.append(backslashes * 2 + 1, L'\\');
      result += L'\"';
      backslashes = 0;
    } else {
      result.append(backslashes, L'\\');
      backslashes = 0;
      result += character;
    }
  }
  result.append(backslashes * 2, L'\\');
  result += L'\"';
  return result;
}
}  // namespace
#endif

bool PlatformProcess::Start(const std::vector<std::string>& arguments,
                            const std::string& working_directory,
                            LogCallback on_log, ExitCallback on_exit,
                            std::string& error) {
  if (arguments.empty() || impl_->running.load()) {
    error = "invalid process start request";
    return false;
  }

#ifdef _WIN32
  SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE read_pipe = nullptr;
  HANDLE write_pipe = nullptr;
  if (!CreatePipe(&read_pipe, &write_pipe, &attributes, 0)) {
    error = "CreatePipe failed";
    return false;
  }
  SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);
  std::wstring command;
  for (const auto& argument : arguments) {
    if (!command.empty()) command += L' ';
    command += QuoteArgument(argument);
  }
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdOutput = write_pipe;
  startup.hStdError = write_pipe;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION information{};
  const std::wstring directory = Utf8ToWide(working_directory);
  if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                      CREATE_NO_WINDOW, nullptr, directory.c_str(), &startup,
                      &information)) {
    CloseHandle(read_pipe);
    CloseHandle(write_pipe);
    error = "CreateProcessW failed with error " + std::to_string(GetLastError());
    return false;
  }
  CloseHandle(write_pipe);
  impl_->process = information.hProcess;
  impl_->thread = information.hThread;
  impl_->output = read_pipe;
#else
  int pipe_handles[2] = {-1, -1};
  if (pipe(pipe_handles) != 0) {
    error = "pipe failed";
    return false;
  }
  const pid_t child = fork();
  if (child < 0) {
    close(pipe_handles[0]);
    close(pipe_handles[1]);
    error = "fork failed";
    return false;
  }
  if (child == 0) {
    close(pipe_handles[0]);
    dup2(pipe_handles[1], STDOUT_FILENO);
    dup2(pipe_handles[1], STDERR_FILENO);
    close(pipe_handles[1]);
    if (!working_directory.empty()) (void)chdir(working_directory.c_str());
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    execv(argv.front(), argv.data());
    _exit(127);
  }
  close(pipe_handles[1]);
  impl_->process = child;
  impl_->output = pipe_handles[0];
#endif

  impl_->running.store(true);
  impl_->reader = std::jthread([this, callback = std::move(on_log)] {
    std::array<char, 1024> buffer{};
    std::string pending;
    while (impl_->running.load()) {
#ifdef _WIN32
      DWORD count = 0;
      if (!ReadFile(impl_->output, buffer.data(), static_cast<DWORD>(buffer.size()), &count, nullptr) || count == 0) break;
      const std::size_t read_count = count;
#else
      const ssize_t count = read(impl_->output, buffer.data(), buffer.size());
      if (count <= 0) break;
      const std::size_t read_count = static_cast<std::size_t>(count);
#endif
      pending.append(buffer.data(), read_count);
      for (std::size_t newline = pending.find('\n'); newline != std::string::npos;
           newline = pending.find('\n')) {
        std::string line = pending.substr(0, newline);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        callback(std::move(line));
        pending.erase(0, newline + 1);
      }
    }
    if (!pending.empty()) callback(std::move(pending));
  });
  impl_->waiter = std::jthread([this, callback = std::move(on_exit)] {
    int exit_code = -1;
#ifdef _WIN32
    WaitForSingleObject(impl_->process, INFINITE);
    DWORD native_code = 0;
    if (GetExitCodeProcess(impl_->process, &native_code)) exit_code = static_cast<int>(native_code);
#else
    int status = 0;
    if (waitpid(impl_->process, &status, 0) >= 0) {
      if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
      else if (WIFSIGNALED(status)) exit_code = 128 + WTERMSIG(status);
    }
#endif
    impl_->running.store(false);
    callback(exit_code);
  });
  return true;
}

void PlatformProcess::Stop() {
  std::scoped_lock stop_lock(impl_->stop_mutex);
  if (impl_->running.exchange(false)) {
#ifdef _WIN32
    if (impl_->process) TerminateProcess(impl_->process, 0);
#else
    if (impl_->process > 0) kill(impl_->process, SIGTERM);
#endif
  }
  if (impl_->waiter.joinable() && impl_->waiter.get_id() != std::this_thread::get_id()) impl_->waiter.join();
  if (impl_->reader.joinable() && impl_->reader.get_id() != std::this_thread::get_id()) impl_->reader.join();
#ifdef _WIN32
  if (impl_->output) CloseHandle(impl_->output);
  if (impl_->thread) CloseHandle(impl_->thread);
  if (impl_->process) CloseHandle(impl_->process);
  impl_->output = nullptr;
  impl_->thread = nullptr;
  impl_->process = nullptr;
#else
  if (impl_->output >= 0) close(impl_->output);
  impl_->output = -1;
  impl_->process = -1;
#endif
}

bool PlatformProcess::IsRunning() const { return impl_->running.load(); }

}  // namespace gem16::studio
