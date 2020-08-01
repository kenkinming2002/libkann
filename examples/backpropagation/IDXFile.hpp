#pragma once

#include <fstream>
#include <vector>

class IDXFile
{
public:
  IDXFile(const char* fileName);

public:
  enum class Type : uint8_t {
    UNSIGNED_BYTE = 0x08,
    SIGNED_BYTE = 0x09,
    SHORT = 0x0B,
    INT = 0x0C,
    FLOAT = 0x0D,
    DOUBLE = 0x0E
  };

public:
  void read(char* s, std::streamsize n);

public:
  auto& file() { return m_file; }
  auto type() { return m_type; }
  auto dimensionCount() { return m_dimensionCount; }
  const auto& dimensions() { return m_dimensions; }

private:
  std::ifstream m_file;
  Type m_type;
  uint8_t m_dimensionCount;
  std::vector<uint32_t> m_dimensions; // Probably overkill, but does not matter
};
