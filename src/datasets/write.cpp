#include <libkann/datasets/write.hpp>

namespace kann
{
  struct [[gnu::packed]] BMPHeader
  {
    char magic[2];
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
  };

  struct [[gnu::packed]] BMPInfoHeader
  {
    uint32_t infoHeaderSize;
    uint16_t width;
    uint16_t height;
    uint16_t colorPlanesCount;
    uint16_t bitsPerPixel;
  };

  void writeImage(std::ostream& os, const Eigen::VectorXd& data, size_t width, size_t height)
  {
    BMPHeader header;
    BMPInfoHeader infoHeader;

    memcpy(header.magic, "BM", 2);
    header.fileSize = sizeof header + sizeof infoHeader + 4 * data.size();
    header.reserved = 0;
    header.dataOffset = sizeof header + sizeof infoHeader;

    infoHeader.infoHeaderSize = sizeof infoHeader;
    infoHeader.width  = width;
    infoHeader.height = height;
    infoHeader.colorPlanesCount = 1;
    infoHeader.bitsPerPixel = 32;

    os.write(reinterpret_cast<const char*>(&header), sizeof header);
    os.write(reinterpret_cast<const char*>(&infoHeader), sizeof infoHeader);
    for(size_t i=0; i<width*height; ++i)
    {
      const uint8_t v = static_cast<uint8_t>(data[i] * 256.0);
      const uint8_t pixel[4] = {v, v, v, v};
      os.write(reinterpret_cast<const char*>(&pixel), sizeof pixel);
    }
  }
}
