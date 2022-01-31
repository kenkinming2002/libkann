#pragma once

#include <libkann/Scope.hpp>

#include <string>

#include <stddef.h>

namespace kann
{
  struct NewParameter
  {
  public:
    NewParameter inScope(Scope scope) const
    {
      return NewParameter{
        .scope = scope + this->scope,
        .name  = this->name,
        .size  = this->size
      };
    }

    std::string qualifiedName() const
    {
      return scope.qualifiedName(name);
    }

  public:
    auto operator<=>(const NewParameter&) const = default;

  public:
    Scope scope;
    std::string name;

    size_t size;
  };

  inline size_t hashCombine(size_t hash1, size_t hash2)
  {
    // Same as in boost::hash_combine. How well does it work? I have no idea.
    return hash1 ^ (hash2 + 0x9e3779b9 + (hash1<<6) + (hash1>>2) );
  }
}


template<>
struct std::hash<kann::NewParameter>
{
  size_t operator()(const kann::NewParameter& parameter) const
  {
    size_t hash1 = std::hash<kann::Scope>{}(parameter.scope);
    size_t hash2 = std::hash<std::string>{}(parameter.name);
    return kann::hashCombine(hash1, hash2);
  }
};
