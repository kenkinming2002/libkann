#include <libkann/datasets/DataSet.hpp>

namespace kann
{
  class RandomDataSet : public DataSet
  {
  public:
    enum Column
    {
      COLUMN_DATA
    };

  public:
    RandomDataSet(size_t dataSize, size_t size);

  public:
    size_t size() const override;

  public:
    Tensor get(size_t column, size_t index) const override;
    double correctness(size_t column, size_t index, const Tensor& data) const override;


  private:
    size_t m_dataSize;
    size_t m_size;
  };
}


