#pragma once

#include <vector>
#include <fstream>

namespace kann
{
  class IDXFile
  {
  public:
    IDXFile(const char* fileName);

  public:
    enum class Type : char {
      UNSIGNED_BYTE = 0x08,
      SIGNED_BYTE = 0x09,
      SHORT = 0x0B,
      INT = 0x0C,
      FLOAT = 0x0D,
      DOUBLE = 0x0E
    };

  public:
    bool read(char* s, std::streamsize n);

  public:
    Type type() const { return m_type; }
    const auto& dimensions() { return m_dimensions; }

  private:
    std::ifstream m_file;

  private:
    Type m_type;
    std::vector<uint32_t> m_dimensions;
  };

}
