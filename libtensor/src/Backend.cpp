#include <libtensor/Backend.hpp>

#include <spdlog/spdlog.h>

#include <filesystem>
#include <sstream>
#include <vector>
#include <string>
#include <optional>

#include <stdlib.h>
#include <dlfcn.h>

namespace tensor
{
  inline static std::string get_environment(const char* name, const char* fallback)
  {
    const char* env = getenv(name);
    return env ? env : fallback;
  }

  inline static std::vector<std::string> split(const std::string& str, char delim)
  {
    std::vector<std::string> tokens;

    std::stringstream ss(str);
    std::string token;
    while(std::getline(ss, token, delim))
      if(!token.empty())
        tokens.push_back(std::move(token));

    return tokens;
  }

  inline static std::optional<Backend> load_backend(const std::string& name)
  {
    static const std::vector<std::string> PATHS = split(get_environment("LIBTENSOR_BACKEND_PATHS", ""), ';');

    void* lib = nullptr;
    for(const std::string& path : PATHS)
    {
      std::filesystem::path file_path = std::filesystem::path(path) / std::filesystem::path(name + ".so");
      spdlog::info("libtensor: Looking at path = {}", file_path.c_str());
      lib = dlopen(file_path.c_str(), RTLD_LAZY);
      if(lib)
        break;

      spdlog::info("libtensor: Failed with error {}", dlerror());
    }

    if(!lib)
    {
      spdlog::warn("libtensor: Failed to load backend with name = {}", name);
      return std::nullopt;
    }

    spdlog::info("libtensor: Successfully loaded backend with name = {} with", name);

    Backend backend;
    backend.sgemm     = (sgemm_t    )dlsym(lib, "sgemm");
    backend.sgecorr2d = (sgecorr2d_t)dlsym(lib, "sgecorr2d");
    backend.sgeconv2d = (sgeconv2d_t)dlsym(lib, "sgeconv2d");

    spdlog::info("libtensor:   .sgemm     = {}", (uintptr_t)backend.sgemm    );
    spdlog::info("libtensor:   .sgecorr2d = {}", (uintptr_t)backend.sgecorr2d);
    spdlog::info("libtensor:   .sgeconv2d = {}", (uintptr_t)backend.sgeconv2d);

    return backend;
  }

  inline static std::vector<Backend> load_backends()
  {
    static const std::vector<std::string> NAMES = split(get_environment("LIBTENSOR_BACKEND_NAMES", ""), ';');

    std::vector<Backend> backends;
    for(const std::string& name : NAMES)
    {
      spdlog::info("libtensor: Loading backend with name = {}", name);
      auto backend = load_backend(name);
      if(backend)
        backends.push_back(*backend);
    }
    return backends;
  }

  std::vector<Backend> backends()
  {
    static const std::vector<Backend> BACKENDS = load_backends();
    return BACKENDS;
  }
}
