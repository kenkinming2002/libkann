#pragma once

#include <vector>
#include <string>

namespace kann
{
  struct Scope
  {
  public:
    static inline const char* SEPERATOR = ".";

  public:
    Scope() = default;
    Scope(std::string name)
    {
      m_names.push_back(std::move(name));
    }

  public:
    friend Scope operator+(Scope lhs, Scope rhs)
    {
      auto result = std::move(lhs);
      result.m_names.insert(result.m_names.end(),
        std::move_iterator(rhs.m_names.begin()),
        std::move_iterator(rhs.m_names.end())
      );
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

  private:
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
