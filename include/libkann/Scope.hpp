#pragma once

#include <libkann/Tag.hpp>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <vector>
#include <string>

namespace kann
{
  struct Scope
  {
  public:
    static inline const char* SEPERATOR = ".";

  public:
    Scope(Tag tag = Tag::ALL)        : m_tag(tag) {}
    Scope(Tag tag, std::string name) : m_tag(tag), m_names{std::move(name)} {}

  public:
    Tag tag() const { return m_tag; }

  public:
    friend Scope operator+(const Scope& lhs, const Scope& rhs)
    {
      Scope result;

      result.m_tag = lhs.m_tag & rhs.m_tag;
      result.m_names.insert(result.m_names.end(), std::move_iterator(lhs.m_names.begin()), std::move_iterator(lhs.m_names.end()));
      result.m_names.insert(result.m_names.end(), std::move_iterator(rhs.m_names.begin()), std::move_iterator(rhs.m_names.end()));

      return result;
    }

  public:
    std::string toString() const
    {
      std::string result;
      for(auto& name : m_names)
      {
        result += name;
        result += ".";
      }
      return result;
    }

    std::string qualifiedName(const std::string& name) const
    {
      return this->toString() + name;
    }

  public:
    auto operator<=>(const Scope&) const = default;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_tag, m_names);
    }

  private:
    Tag m_tag;
    std::vector<std::string> m_names;
  };
}

template<>
struct std::hash<kann::Scope>
{
  size_t operator()(const kann::Scope& scope) const
  {
    return std::hash<std::string>{}(scope.toString());
  }
};
