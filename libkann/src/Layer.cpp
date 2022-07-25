#include <libkann/Layer.hpp>

#include <libtensor/SL.hpp>

#include <filesystem>
#include <fstream>

#include <fmt/core.h>

namespace kann
{
  void Layer::save_parameters(const std::string& dirname, bool include_gradient) const
  {
    std::filesystem::create_directories(dirname);
    for(const auto& [name, parameter] : parameters_map())
    {
      tensor::save_tensor(parameter->value,    fmt::format("{}/{}.value",    dirname, name));
      if(!include_gradient) continue;
      tensor::save_tensor(parameter->gradient, fmt::format("{}/{}.gradient", dirname, name));
    }
  }

  void Layer::load_parameters(const std::string& dirname, bool include_gradient)
  {
    for(const auto& [name, parameter] : parameters_map())
    {
      parameter->value    = tensor::load_tensor(fmt::format("{}/{}.value",    dirname, name));
      if(!include_gradient) continue;
      parameter->gradient = tensor::load_tensor(fmt::format("{}/{}.gradient", dirname, name));
    }
  }
}
