#pragma once

#include <string>

#include <stddef.h>

namespace kann
{
  struct NewParameter
  {
  public:
    NewParameter withPrefix(std::string prefix)
    {
      static const char* SEPERATOR = ".";
      return NewParameter{
        .name = prefix+SEPERATOR+name,
        .size = size
      };
    }

  public:
    template<typename Archive>
    void serialize(Archive& archive) const
    {
      archive(name);
      archive(size);
    }

  public:
    std::string name;
    size_t size;
  };
}
