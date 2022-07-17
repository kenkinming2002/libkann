#pragma once

#include <libtensor/Shape.hpp>
#include <libtensor/Buffer.hpp>

#include <range/v3/all.hpp>

#include <random>
#include <memory>
#include <tuple>

namespace tensor
{
  template<typename T>
  struct Tensor
  {
  public:
    Shape shape;
    std::shared_ptr<const Buffer<T>> buffer;

  public:
    Tensor() = default;
    Tensor(Shape shape, std::shared_ptr<const Buffer<T>> buffer) : shape(std::move(shape)), buffer(std::move(buffer)) {}

  public:
    Tensor reshape(Shape new_shape) const
    {
      assert(shape.size() == new_shape.size());
      return Tensor(std::move(new_shape), buffer);
    }

    Tensor flatten() const { return reshape(Shape::make(shape.size())); }
    Tensor flatten(const auto&... hints)   const requires(sizeof...(hints)>0) { return reshape(shape.flatten(hints...)); }
    Tensor unflatten(const auto&... hints) const requires(sizeof...(hints)>0) { return reshape(shape.unflatten(hints...)); }
  };
}
