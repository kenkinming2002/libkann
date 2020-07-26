#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

class CSVFile
{
public:
  CSVFile(const char* name) : m_file(name)
  {
    if(!m_file)
      throw std::runtime_error(std::string("Failed to open csv file ") + name + "for writing");
  }

public:
  template<typename Arg, typename... Args>
  void write(const Arg& arg, const Args&... args)
  {
    m_file << arg;
    ((m_file << "," << args), ...);
    m_file << '\n';
  }

private:
  std::fstream m_file;
};
