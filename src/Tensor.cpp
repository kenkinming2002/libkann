#include <libkann/Tensor.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  Tensor Tensor::reduce(std::vector<const Tensor*> values)
  {
    assert(!values.empty());

    size_t size = values.front()->size();
    assert(ranges::all_of(values, [size](const Tensor* value) { return value->size() == size; }));

    Tensor result(size);
    result.asArray() = values.front()->asArray();
    for(const Tensor* value : values | ranges::views::drop(1))
      result.asArray() += value->asArray();

    return result;
  }

  // Half-opened
  static inline bool between(Vec2 vec, Vec2 lower, Vec2 upper)
  {
    return (lower.x() <= vec.x() && vec.x() < upper.x())
        && (lower.y() <= vec.y() && vec.y() < upper.y());
  }

  static inline auto pad(const Eigen::Ref<const Eigen::MatrixXd> input, Vec2 padding_size)
  {
    Vec2 input_size  = Vec2(input.cols(), input.rows());
    Vec2 output_size = input_size + 2 * padding_size;
    return Eigen::MatrixXd::NullaryExpr(output_size.height(), output_size.width(), [=](Eigen::Index row, Eigen::Index col){
      Vec2 pos = Vec2(col, row);
      if(between(pos, padding_size, input_size + padding_size))
      {
        Vec2 real_pos = pos - padding_size;
        return input(real_pos.y(), real_pos.x());
      }
      return 0.0;
    });
  }

  Tensor Tensor::convolve(const Tensor& _input, const Tensor& _kernel, Vec2 input_size, Vec2 output_size, Vec2 kernel_size)
  {
    Vec2 padding_size = output_size + (kernel_size - Vec2(1,1)) - input_size;
    assert(padding_size % 2 == Vec2(0,0));
    padding_size = padding_size / 2;

    Tensor _output(output_size.height() * output_size.width());
    auto input  = pad(_input.asMatrix(input_size.height(), input_size.width()), padding_size);
    auto output = _output.asMatrix(output_size.height(), output_size.width());
    auto kernel = _kernel.asMatrix(kernel_size.height(), kernel_size.width()).reverse();
    output = Eigen::MatrixXd::NullaryExpr(output_size.height(), output_size.width(), [&input, &kernel](Eigen::MatrixXd::Index rows, Eigen::MatrixXd::Index cols){
      return input.block(rows, cols, kernel.rows(), kernel.cols()).cwiseProduct(kernel).sum();
    });
    return _output;
  }

  Tensor Tensor::cross_correlate(const Tensor& _input, const Tensor& _kernel, Vec2 input_size, Vec2 output_size, Vec2 kernel_size)
  {
    Vec2 padding_size = output_size + (kernel_size - Vec2(1,1)) - input_size;
    assert(padding_size % 2 == Vec2(0,0));
    padding_size = padding_size / 2;

    Tensor _output(output_size.height() * output_size.width());
    auto input  = pad(_input.asMatrix(input_size.height(), input_size.width()), padding_size);
    auto output = _output.asMatrix(output_size.height(), output_size.width());
    auto kernel = _kernel.asMatrix(kernel_size.height(), kernel_size.width());
    output = Eigen::MatrixXd::NullaryExpr(output_size.height(), output_size.width(), [&input, &kernel](Eigen::MatrixXd::Index rows, Eigen::MatrixXd::Index cols){
        return input.block(rows, cols, kernel.rows(), kernel.cols()).cwiseProduct(kernel).sum();
    });
    return _output;
  }

  Tensor Tensor::concat(std::vector<const Tensor*> values, size_t size, size_t count)
  {
    Tensor result(size * count);

    assert(values.size() == count);
    for(const auto& [i, value] : ranges::views::enumerate(values))
      result.asArray().segment(i * size, size) = value->asArray();

    return result;
  }

  std::vector<Tensor> Tensor::split(const Tensor& value, size_t size, size_t count)
  {
    return ranges::views::ints(0uz, count) | ranges::views::transform([&](size_t i) {
      Tensor result(size);
      result.asArray() = value.asArray().segment(i * size, size);
      return result;
    }) | ranges::to_vector;
  }
}
