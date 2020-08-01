#include "IDXFile.hpp"

#include <stdexcept>
#include <cassert>

namespace
{
  // TODO: support non-gcc
  void byteSwap(uint32_t& x)
  {
    x = __builtin_bswap32(x);
  }
}

IDXFile::IDXFile(const char* fileName) : m_file(fileName, std::ifstream::binary)
{
  if(!m_file)
    throw std::runtime_error(std::string("Failed to open csv file ") + fileName + "for writing");

  m_file.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);

  // Magic Number
  {
    std::byte magicNumber[4];
    m_file.read(reinterpret_cast<char*>(&magicNumber), sizeof magicNumber);
    if(magicNumber[0] != std::byte(0) || magicNumber[1] != std::byte(0))
      throw std::runtime_error(std::string("Invalid magic number") + fileName);

    m_type = static_cast<Type>(magicNumber[2]);
    m_dimensionCount = static_cast<uint8_t>(magicNumber[3]);
  }

  // Dimensions
  {
    m_dimensions.resize(m_dimensionCount);
    m_file.read(reinterpret_cast<char*>(m_dimensions.data()), m_dimensions.size() * sizeof(uint32_t));
    for(uint32_t& dimension: m_dimensions)
      byteSwap(dimension);
  }
}

void IDXFile::read(char* s, std::streamsize n)
{
  m_file.read(s, n);
}
