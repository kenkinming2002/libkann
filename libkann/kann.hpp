#pragma once

#include <iosfwd>
#include <string>

#include <libkann/export.hpp>

namespace kann
{
  // Print a greeting for the specified name into the specified
  // stream. Throw std::invalid_argument if the name is empty.
  //
  LIBKANN_SYMEXPORT void
  say_hello (std::ostream&, const std::string& name);
}
