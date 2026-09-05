#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

#include <httplib.h>

namespace gem16::server {

inline bool IsLoopbackHost(std::string_view host) {
  return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

// No CORS wildcard: native SDKs have no Origin; browsers must be same-origin.
inline std::string ValidateLocalHttpRequest(const httplib::Request& request,
                                           int port) {
  const auto host = request.get_header_value("Host");
  const auto suffix = port == 80 ? std::string() : ":" + std::to_string(port);
  if (request.get_header_value_count("Host") != 1U ||
      (host != "127.0.0.1" + suffix && host != "localhost" + suffix &&
       host != "[::1]" + suffix)) return "Host must identify the local server";
  if (request.has_header("Origin") &&
      (request.get_header_value_count("Origin") != 1U ||
       request.get_header_value("Origin") != "http://" + host))
    return "cross-origin browser requests are unsupported";
  if (request.method == "POST" &&
      (request.path == "/v1/chat/completions" || request.path == "/v1/responses")) {
    auto mime = request.get_header_value("Content-Type");
    mime = mime.substr(0, mime.find(';'));
    while (!mime.empty() && mime.back() == ' ') mime.pop_back();
    std::transform(mime.begin(), mime.end(), mime.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (request.get_header_value_count("Content-Type") != 1U || mime != "application/json")
      return "generation requests require Content-Type: application/json";
  }
  return {};
}

}  // namespace gem16::server
