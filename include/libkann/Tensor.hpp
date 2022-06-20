#pragma once

#include <libkann/Shape.hpp>

#include <memory>

namespace kann
{
  /* A storage could be
   *
   * 1: MutableRefStorage
   * 2: RefStorage
   * 1: SharedStorage
   * 4: ExclusiveStorage
   *
   * where the convesion diagram is
   *    Storage     ->    RefStorage
   *        ^                  ^
   *        |                  |
   * MutableStorage -> MutableRefStorage
   *
   * To facilitate conversion, there are possibly 2 method on each storage:
   * 1: as_const()
   * 2: as_ref()
   *
   * A storage of given size could also be
   * created for MutableStorage via member method create(size_t). While it may
   * be tempting to create the same method for Storage, creating storage and
   * then not being able to modify it is rather pointness, which is why it is
   * not provided. */

  template<typename StorageType>
  struct TensorBase
  {
  public:
    TensorBase(StorageType storage, size_t offset, Shape shape)
      : m_storage(std::move(storage)), m_offset(std::move(offset)), m_shape(std::move(shape)) {}

  public:
    static TensorBase create(Shape shape)
    {
      size_t size = shape.size();
      return TensorBase(StorageType::create(size), 0, std::move(shape));
    }

  public:
    // Explicit namespace qualification to subvert injected class name to use
    // class template argument deduction
    auto as_const() { return kann::TensorBase(m_storage.as_const(), m_offset, m_shape); }
    auto as_ref() const { return kann::TensorBase(m_storage.as_ref(), m_offset, m_shape); }

  // Indexing and reshaping operation
  public:
    TensorBase operator[](size_t i) const
    {
      Shape new_shape = m_shape.split(1, m_shape.rank() - 1).second;
      size_t stride = new_shape.size();
      return TensorBase(m_storage, m_offset + i * stride, std::move(new_shape));
    }

    TensorBase reshape(Shape new_shape) const
    {
      assert(m_shape.size() == new_shape.size());
      return TensorBase(m_storage, m_offset, new_shape);
    }

  public:
    auto data() const { return m_storage.data() + m_offset; }

    auto dimension(size_t i) const { return m_shape.dimension(i); }
    auto rank() const { return m_shape.rank(); }
    auto size() const { return m_shape.size(); }

  public:
    auto& get(size_t i) const { assert(i<size()); return data()[i]; }

  public:
    bool is_scalar() const { return m_shape.is_scalar(); }
    bool is_vector() const { return m_shape.is_vector(); }
    bool is_matrix() const { return m_shape.is_matrix(); }

  public:
    const Shape& shape() const { return m_shape; }

  public:
    void fill(float value)
    {
      ranges::fill_n(data(), size(), value);
    }

    template<typename PRNG>
    void fill_normal(PRNG& prng, float mean, float stddev)
    {
      std::normal_distribution<float> dist(mean, stddev);
      ranges::generate_n(data(), size(), [&]() { return dist(prng); });
    }

  public:
    static TensorBase constant(Shape shape, float value)
    {
      TensorBase result = TensorBase::create(std::move(shape));
      result.fill(value);
      return result;
    }

    template<typename PRNG>
    static TensorBase normal(Shape shape, PRNG& prng, float mean, float stddev)
    {
      TensorBase result = TensorBase::create(std::move(shape));
      result.fill_normal(prng, mean, stddev);
      return result;
    }

  private:
    StorageType m_storage;

  private:
    size_t m_offset;
    Shape m_shape;
  };

  struct RefStorage
  {
  public:
    RefStorage(const float* data, size_t size) : m_data(data), m_size(size) {}

  public:
    const float* data() const { return m_data; }
    size_t size() const { return m_size; }

  public:
    RefStorage as_const() const { return *this; }
    RefStorage as_ref() const { return *this; }

  private:
    const float* m_data;
    size_t m_size;
  };

  struct MutableRefStorage
  {
  public:
    MutableRefStorage(float* data, size_t size) : m_data(data), m_size(size) {}

  public:
    float* data() const { return m_data; }
    size_t size() const { return m_size; }

  public:
    RefStorage as_const() const { return RefStorage(m_data, m_size); }
    MutableRefStorage as_ref() const { return *this; }

  private:
    float* m_data;
    size_t m_size;
  };

  struct Storage
  {
  public:
    Storage(std::shared_ptr<const float[]> data, size_t size) : m_data(std::move(data)), m_size(size) {}

  public:
    const float* data() const { return m_data.get(); }
    size_t size() const { return m_size; }

  public:
    Storage as_const() const { return *this; }
    RefStorage as_ref() const { return RefStorage(data(), size()); }

  private:
    std::shared_ptr<const float[]> m_data;
    size_t m_size;
  };

  struct MutableStorage
  {
  public:
    MutableStorage(std::shared_ptr<float[]> data, size_t size) : m_data(std::move(data)), m_size(size) {}

  public:
    float* data() const { return m_data.get(); }
    size_t size() const { return m_size; }

  public:
    Storage as_const() const { return Storage(m_data, m_size); }
    MutableRefStorage as_ref() const { return MutableRefStorage(data(), size()); }

  public:
    static MutableStorage create(size_t size)
    {
      // make_shared_for_overwrite is available only for gcc 12
      return MutableStorage(std::make_unique_for_overwrite<float[]>(size), size);
    }

  private:
    std::shared_ptr<float[]> m_data;
    size_t m_size;
  };

  using MutableTensorRef = TensorBase<MutableRefStorage>;
  using TensorRef        = TensorBase<RefStorage>;
  using MutableTensor    = TensorBase<MutableStorage>;
  using Tensor           = TensorBase<Storage>;
}
