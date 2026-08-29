#include "media_loader.h"

#include "platform_process.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <random>
#include <unordered_map>

namespace gem16::studio {
namespace {

struct MediaType {
  MediaKind kind;
  const char* mime;
  const char* format;
};

const std::unordered_map<std::string, MediaType> kMediaTypes{
    {"png", {MediaKind::kImage, "image/png", "png"}},
    {"jpg", {MediaKind::kImage, "image/jpeg", "jpeg"}},
    {"jpeg", {MediaKind::kImage, "image/jpeg", "jpeg"}},
    {"bmp", {MediaKind::kImage, "image/bmp", "bmp"}},
    {"wav", {MediaKind::kAudio, "audio/wav", "wav"}},
    {"flac", {MediaKind::kAudio, "audio/flac", "flac"}},
    {"mp3", {MediaKind::kAudio, "audio/mpeg", "mp3"}},
};

bool IsTextExtension(const std::string& extension) {
  static const std::unordered_map<std::string, bool> extensions{
      {"txt", true}, {"md", true}, {"markdown", true}, {"csv", true},
      {"tsv", true}, {"json", true}, {"jsonl", true}, {"xml", true},
      {"yaml", true}, {"yml", true}, {"log", true}, {"kt", true},
      {"kts", true}, {"java", true}, {"py", true}, {"c", true},
      {"cc", true}, {"cpp", true}, {"h", true}, {"hpp", true},
      {"js", true}, {"jsx", true}, {"ts", true}, {"tsx", true},
      {"html", true}, {"htm", true}, {"css", true}, {"sql", true},
      {"toml", true}, {"ini", true}, {"properties", true},
  };
  return extensions.contains(extension);
}

std::string Extension(const std::filesystem::path& path) {
  std::string value = path.extension().string();
  if (!value.empty() && value.front() == '.') value.erase(value.begin());
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool ValidUtf8(std::string_view text) {
  for (std::size_t index = 0; index < text.size();) {
    const auto lead = static_cast<unsigned char>(text[index]);
    std::size_t count = 0;
    if (lead < 0x80U) count = 1;
    else if ((lead & 0xE0U) == 0xC0U) count = 2;
    else if ((lead & 0xF0U) == 0xE0U) count = 3;
    else if ((lead & 0xF8U) == 0xF0U) count = 4;
    else return false;
    if (index + count > text.size()) return false;
    for (std::size_t offset = 1; offset < count; ++offset) {
      if ((static_cast<unsigned char>(text[index + offset]) & 0xC0U) != 0x80U)
        return false;
    }
    index += count;
  }
  return true;
}

std::optional<std::filesystem::path> FindExecutable(const char* name) {
  const char* path_value = std::getenv("PATH");
  if (!path_value) return std::nullopt;
#ifdef _WIN32
  constexpr char separator = ';';
  const std::string executable = std::string(name) + ".exe";
#else
  constexpr char separator = ':';
  const std::string executable = name;
#endif
  std::string_view paths(path_value);
  while (!paths.empty()) {
    const auto end = paths.find(separator);
    const auto directory = paths.substr(0, end);
    if (!directory.empty()) {
      const auto candidate = std::filesystem::path(directory) / executable;
      if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    if (end == std::string_view::npos) break;
    paths.remove_prefix(end + 1);
  }
  return std::nullopt;
}

bool ExtractPdf(const std::filesystem::path& source, std::string& text,
                std::string& error) {
  const auto pdftotext = FindExecutable("pdftotext");
  if (!pdftotext) {
    error = "PDF text extraction requires the optional pdftotext utility. "
            "Install Poppler or attach exported text.";
    return false;
  }
  std::filesystem::path temporary_directory;
  std::random_device random;
  for (int attempt = 0; attempt < 32 && temporary_directory.empty(); ++attempt) {
    const auto candidate = std::filesystem::temp_directory_path() /
        ("gem16-pdf-" + std::to_string(random()) + '-' + std::to_string(random()));
    std::error_code directory_error;
    if (std::filesystem::create_directory(candidate, directory_error)) {
#ifndef _WIN32
      std::filesystem::permissions(candidate, std::filesystem::perms::owner_all,
                                   std::filesystem::perm_options::replace,
                                   directory_error);
      if (directory_error) {
        std::filesystem::remove(candidate);
        continue;
      }
#endif
      temporary_directory = candidate;
    }
  }
  if (temporary_directory.empty()) {
    error = "Could not create a private temporary PDF extraction directory";
    return false;
  }
  const auto destination = temporary_directory / "document.txt";
  PlatformProcess process;
  std::mutex mutex;
  std::condition_variable condition;
  bool finished = false;
  int exit_code = -1;
  std::string diagnostic;
  std::string start_error;
  if (!process.Start({pdftotext->string(), "-enc", "UTF-8", "-nopgbrk",
                      source.string(), destination.string()},
                     source.parent_path().string(),
                     [&](std::string line) {
                       if (diagnostic.size() < 1024U) diagnostic += std::move(line) + '\n';
                     },
                     [&](int code) {
                       std::lock_guard lock(mutex);
                       exit_code = code;
                       finished = true;
                       condition.notify_all();
                     }, start_error)) {
    error = "Could not start pdftotext: " + start_error;
    std::filesystem::remove_all(temporary_directory);
    return false;
  }
  constexpr std::uintmax_t maximum_extracted_bytes =
      kMaxDocumentCharacters * 4U;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  bool extraction_limit_reached = false;
  while (true) {
    {
      std::unique_lock lock(mutex);
      if (condition.wait_for(lock, std::chrono::milliseconds(100),
                             [&] { return finished; }))
        break;
    }
    std::error_code size_error;
    const auto output_size = std::filesystem::is_regular_file(destination, size_error)
                                 ? std::filesystem::file_size(destination, size_error)
                                 : 0U;
    if ((!size_error && output_size > maximum_extracted_bytes) ||
        std::chrono::steady_clock::now() >= deadline) {
      extraction_limit_reached = true;
      process.Stop();
      break;
    }
  }
  process.Stop();
  if (extraction_limit_reached) {
    std::filesystem::remove_all(temporary_directory);
    error = "PDF extraction exceeded the 30 second or 500,000 character safety limit.";
    return false;
  }
  if (exit_code != 0) {
    std::filesystem::remove_all(temporary_directory);
    error = "PDF extraction failed" +
            (diagnostic.empty() ? std::string{} : ": " + diagnostic.substr(0, 500));
    return false;
  }
  std::error_code output_error;
  const auto output_size = std::filesystem::file_size(destination, output_error);
  if (output_error || output_size > maximum_extracted_bytes) {
    std::filesystem::remove_all(temporary_directory);
    error = "Extracted PDF text exceeds the 500,000 character safety limit.";
    return false;
  }
  std::ifstream input(destination, std::ios::binary);
  text.assign(std::istreambuf_iterator<char>(input), {});
  std::error_code remove_error;
  std::filesystem::remove_all(temporary_directory, remove_error);
  if (input.bad() || text.empty() || !ValidUtf8(text)) {
    error = "PDF contains no extractable UTF-8 text; scanned PDFs require OCR.";
    return false;
  }
  if (text.size() > kMaxDocumentCharacters) {
    error = "Extracted PDF text exceeds the 500,000 character limit.";
    return false;
  }
  return true;
}

}  // namespace

std::string EncodeBase64(const std::vector<std::uint8_t>& bytes) {
  constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.reserve(4U * ((bytes.size() + 2U) / 3U));
  for (std::size_t index = 0; index < bytes.size(); index += 3U) {
    const std::uint32_t a = bytes[index];
    const std::uint32_t b = index + 1U < bytes.size() ? bytes[index + 1U] : 0U;
    const std::uint32_t c = index + 2U < bytes.size() ? bytes[index + 2U] : 0U;
    const std::uint32_t value = (a << 16U) | (b << 8U) | c;
    result.push_back(alphabet[(value >> 18U) & 63U]);
    result.push_back(alphabet[(value >> 12U) & 63U]);
    result.push_back(index + 1U < bytes.size() ? alphabet[(value >> 6U) & 63U] : '=');
    result.push_back(index + 2U < bytes.size() ? alphabet[value & 63U] : '=');
  }
  return result;
}

bool LoadMediaAttachment(const std::filesystem::path& path,
                         MediaAttachment& attachment, std::string& error) {
  error.clear();
  std::error_code filesystem_error;
  const auto normalized = std::filesystem::absolute(path, filesystem_error).lexically_normal();
  if (filesystem_error || !std::filesystem::is_regular_file(normalized, filesystem_error)) {
    error = "Attachment does not exist: " + path.string();
    return false;
  }
  const std::uint64_t size = std::filesystem::file_size(normalized, filesystem_error);
  if (filesystem_error || size == 0U || size > kMaxSingleAttachmentBytes) {
    error = size > kMaxSingleAttachmentBytes
                ? "Attachment exceeds the 10 MiB per-file limit: " + normalized.filename().string()
                : "Attachment is empty or unreadable: " + normalized.filename().string();
    return false;
  }
  const std::string extension = Extension(normalized);
  if (extension == "pdf") {
    std::string text;
    if (!ExtractPdf(normalized, text, error)) return false;
    attachment = {};
    attachment.kind = MediaKind::kDocument;
    attachment.file_name = normalized.filename().string();
    attachment.mime_type = "application/pdf";
    attachment.format = "pdf";
    attachment.document_text = std::move(text);
    attachment.byte_size = size;
    return true;
  }
  std::ifstream input(normalized, std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  if (input.bad() || bytes.size() != size) {
    error = "Attachment could not be read completely: " + normalized.filename().string();
    return false;
  }
  attachment = {};
  attachment.file_name = normalized.filename().string();
  attachment.byte_size = size;
  attachment.format = extension;
  if (IsTextExtension(extension)) {
    std::string text(bytes.begin(), bytes.end());
    if (text.starts_with("\xEF\xBB\xBF")) text.erase(0, 3);
    if (!ValidUtf8(text)) {
      error = "Text attachment is not valid UTF-8: " + attachment.file_name;
      return false;
    }
    text.erase(std::remove(text.begin(), text.end(), '\0'), text.end());
    if (text.empty() || text.size() > kMaxDocumentCharacters) {
      error = text.empty() ? "Text attachment contains no readable text"
                           : "Text attachment exceeds the 500,000 character limit";
      return false;
    }
    attachment.kind = MediaKind::kDocument;
    attachment.mime_type = "text/plain";
    attachment.document_text = std::move(text);
    return true;
  }
  const auto type = kMediaTypes.find(extension);
  if (type == kMediaTypes.end()) {
    error = "Unsupported file type. Use text, PNG, JPEG, BMP, WAV, FLAC, or MP3.";
    return false;
  }
  attachment.kind = type->second.kind;
  attachment.mime_type = type->second.mime;
  attachment.format = type->second.format;
  attachment.bytes = std::move(bytes);
  return true;
}

}  // namespace gem16::studio
