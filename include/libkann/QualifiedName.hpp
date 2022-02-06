#pragma once

#include <libkann/Scope.hpp>

#include <cereal/types/string.hpp>

#include <string>

#include <stddef.h>

namespace kann
{
  struct QualifiedName
  {
  public:
    std::string toString() const
    {
      return scope.toString() + name;
    }

  public:
    QualifiedName derive(const std::string& suffix) const
    {
      return QualifiedName{
        .scope = scope,
        .name = name + "_" + suffix,
        .size = size,
      };
    }

    QualifiedName derive(const std::string& suffix, size_t size) const
    {
      return QualifiedName{
        .scope = scope,
        .name = name + "_" + suffix,
        .size = size,
      };
    }

  public:
    auto operator<=>(const QualifiedName&) const = default;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(scope, name, size);
    }

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
struct std::hash<kann::QualifiedName>
{
  size_t operator()(const kann::QualifiedName& qualifiedName) const
  {
    size_t hash1 = std::hash<kann::Scope>{}(qualifiedName.scope);
    size_t hash2 = std::hash<std::string>{}(qualifiedName.name);
    return kann::hashCombine(hash1, hash2);
  }
};
