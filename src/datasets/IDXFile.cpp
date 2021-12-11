#include <libkann/datasets/IDXFile.hpp>

// TODO: Where do we find this header on windows?
#include <arpa/inet.h>

namespace kann
{
  IDXFile::IDXFile(const char* fileName)
  {
    m_file.open(fileName);
    if(!m_file)
      throw std::runtime_error(std::string("Failed to open file ") + fileName);

    char magic[4];
    if(!m_file.read(magic, sizeof magic))
      throw std::runtime_error(std::string("Invalid file format ") + fileName);

    if(magic[0] != 0 || magic[1] != 0)
      throw std::runtime_error(std::string("Invalid file format ") + fileName);

    m_type = static_cast<Type>(magic[2]);
    m_dimensions.resize(magic[3]);

    uint32_t* data = m_dimensions.data();
    size_t    size = m_dimensions.size();
    if(!m_file.read(reinterpret_cast<char*>(data), size * sizeof data[0]))
      throw std::runtime_error(std::string("Invalid file format ") + fileName);

    for(auto& dimension: m_dimensions)
      dimension = ntohl(dimension);
  }

  bool IDXFile::read(char* s, std::streamsize n)
  {
    if(!m_file.read(s, n))
      return false;

    return true;
  }
}
