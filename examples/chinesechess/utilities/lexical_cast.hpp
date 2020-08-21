#pragma once

#include <sstream>

template<typename T>
T lexical_cast(const char* str)
{
  T t;

  std::stringstream ss;
  ss << str;
  ss >> t;
  if(ss.fail() || !ss.eof())
    throw std::invalid_argument(str);

  return t;
}
