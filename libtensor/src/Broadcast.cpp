#include <libtensor/Broadcast.hpp>

namespace tensor
{
#define BROADCAST_OUTER(op_name, op)                                   \
  template<typename T>                                                 \
  Tensor<T> broadcast_##op_name##_outer(Tensor<T> a, Tensor<T> b)      \
  {                                                                    \
    const Shape result_shape = a.shape;                                \
    const Shape outer_shape  = result_shape.drop_back(b.shape.rank()); \
    const Shape inner_shape  = result_shape.back(b.shape.rank());      \
    assert(b.shape == inner_shape);                                    \
                                                                       \
    const size_t M = outer_shape.size();                               \
    const size_t N = inner_shape.size();                               \
                                                                       \
    auto buffer_a = a.buffer;                                          \
    auto buffer_b = b.buffer;                                          \
    auto buffer_c = std::make_shared<Buffer<T>>(M*N);                  \
                                                                       \
    for(size_t m=0; m<M; ++m)                                          \
      for(size_t n=0; n<N; ++n)                                        \
        (*buffer_c)[m*N+n] = (*buffer_a)[m*N+n] op (*buffer_b)[n];     \
                                                                       \
    return Tensor<T>(result_shape, std::move(buffer_c));               \
  }

  BROADCAST_OUTER(add, +);
  BROADCAST_OUTER(sub, -);
  BROADCAST_OUTER(mul, *);
  BROADCAST_OUTER(div, /);

#define BROADCAST_INNER(op_name, op)                                    \
  template<typename T>                                                  \
  Tensor<T> broadcast_##op_name##_inner(Tensor<T> a, Tensor<T> b)       \
  {                                                                     \
    const Shape result_shape = a.shape;                                 \
    const Shape outer_shape  = result_shape.front(b.shape.rank());      \
    const Shape inner_shape  = result_shape.drop_front(b.shape.rank()); \
    assert(b.shape == outer_shape);                                     \
                                                                        \
    const size_t M = outer_shape.size();                                \
    const size_t N = inner_shape.size();                                \
                                                                        \
    auto buffer_a = a.buffer;                                           \
    auto buffer_b = b.buffer;                                           \
    auto buffer_c = std::make_shared<Buffer<T>>(M*N);                   \
                                                                        \
    for(size_t m=0; m<M; ++m)                                           \
      for(size_t n=0; n<N; ++n)                                         \
        (*buffer_c)[m*N+n] = (*buffer_a)[m*N+n] op (*buffer_b)[m];      \
                                                                        \
    return Tensor<T>(result_shape, std::move(buffer_c));                \
  }

  BROADCAST_INNER(add, +);
  BROADCAST_INNER(sub, -);
  BROADCAST_INNER(mul, *);
  BROADCAST_INNER(div, /);

#define BROADCAST_INSTANTIATE_TYPED(T, op_name, dir) \
  template Tensor<T> broadcast_##op_name##_##dir(Tensor<T> a, Tensor<T> b); \

#define BROADCAST_INSTANTIATE(op_name, dir) \
  BROADCAST_INSTANTIATE_TYPED(float,       op_name, dir)   \
  BROADCAST_INSTANTIATE_TYPED(double,      op_name, dir) \
  BROADCAST_INSTANTIATE_TYPED(long double, op_name, dir)

  BROADCAST_INSTANTIATE(add, outer)
  BROADCAST_INSTANTIATE(sub, outer)
  BROADCAST_INSTANTIATE(mul, outer)
  BROADCAST_INSTANTIATE(div, outer)
  BROADCAST_INSTANTIATE(add, inner)
  BROADCAST_INSTANTIATE(sub, inner)
  BROADCAST_INSTANTIATE(mul, inner)
  BROADCAST_INSTANTIATE(div, inner)
}
