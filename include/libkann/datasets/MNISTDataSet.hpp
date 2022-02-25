#include <libkann/datasets/DataSet.hpp>

#include <libkann/datasets/IDXFile.hpp>

#include <fstream>

namespace kann
{
  class MNISTDataSet : public DataSet
  {
  public:
    enum Column
    {
      COLUMN_IMAGE,
      COLUMN_LABEL
    };

  public:
    static constexpr size_t IMAGE_WIDTH = 28;
    static constexpr size_t IMAGE_SIZE  = IMAGE_WIDTH * IMAGE_WIDTH;

  public:
    MNISTDataSet(const char* imageFileName, const char* labelFileName);

  public:
    size_t size() const override;

  public:
    CRef<Tensor> get(size_t column, size_t index) const override;
    double correctness(size_t column, size_t index, const Tensor& data) const override;

  private:
    struct Data
    {
      uint8_t image[IMAGE_SIZE];
      uint8_t label;
    }; // TODO: How to pack the struct better
    std::vector<Data> m_data;
  };
}
