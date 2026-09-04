#pragma once
#include <string>

#include "types.h"
namespace gem16::studio {
bool UsesCatalogPaths(const ServerConfig& config);
// Restore only a currently qualified lock identity; never infer a revision from
// a path.
ServerConfig SavedServerSelection(const std::string& record,
                                  const ServerConfig& local);
}  // namespace gem16::studio
