#include <libtensor/Broadcast.hpp>

namespace tensor
{
#define BROADCAST_OUTER_RAW(op_name, op)                                                                                                               \
  template<typename T>                                                                                                                                 \
  static inline void broadcast_##op_name##_outer_raw(size_t M, size_t N, const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C) noexcept \
  {                                                                                                                                                    \
    for(size_t m=0; m<M; ++m)                                                                                                                          \
      for(size_t n=0; n<N; ++n)                                                                                                                        \
        C[m*N+n] = A[m*N+n] op B[n];                                                                                                                   \
  }

  BROADCAST_OUTER_RAW(add, +);
  BROADCAST_OUTER_RAW(sub, -);
  BROADCAST_OUTER_RAW(mul, *);
  BROADCAST_OUTER_RAW(div, /);

#define BROADCAST_INNER_RAW(op_name, op)                                                                                                               \
  template<typename T>                                                                                                                                 \
  static inline void broadcast_##op_name##_inner_raw(size_t M, size_t N, const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C) noexcept \
  {                                                                                                                                                    \
    for(size_t m=0; m<M; ++m)                                                                                                                          \
      for(size_t n=0; n<N; ++n)                                                                                                                        \
        C[m*N+n] = A[m*N+n] op B[m];                                                                                                                   \
  }

  BROADCAST_INNER_RAW(add, +);
  BROADCAST_INNER_RAW(sub, -);
  BROADCAST_INNER_RAW(mul, *);
  BROADCAST_INNER_RAW(div, /);

#define BROADCAST_OUTER(op_name)                                  \
  template<typename T>                                            \
  Tensor<T> broadcast_##op_name##_outer(Tensor<T> a, Tensor<T> b) \
  {                                                               \
    size_t M = a.shape.dimension(0);                              \
    size_t N = a.shape.dimension(1);                              \
    assert(b.shape.dimension(0) == N);                            \
                                                                  \
    auto buffer_a = a.buffer;                                     \
    auto buffer_b = b.buffer;                                     \
    auto buffer_c = std::make_shared<Buffer<T>>(M*N);             \
    broadcast_##op_name##_outer_raw(M, N,                         \
      buffer_a->data().data(),                                    \
      buffer_b->data().data(),                                    \
      buffer_c->data().data()                                     \
    );                                                            \
    return Tensor<T>(Shape::make(M, N), std::move(buffer_c));     \
  }

  BROADCAST_OUTER(add);
  BROADCAST_OUTER(sub);
  BROADCAST_OUTER(mul);
  BROADCAST_OUTER(div);

#define BROADCAST_INNER(op_name)                                  \
  template<typename T>                                            \
  Tensor<T> broadcast_##op_name##_inner(Tensor<T> a, Tensor<T> b) \
  {                                                               \
    size_t M = a.shape.dimension(0);                              \
    size_t N = a.shape.dimension(1);                              \
    assert(b.shape.dimension(0) == M);                            \
                                                                  \
    auto buffer_a = a.buffer;                                     \
    auto buffer_b = b.buffer;                                     \
    auto buffer_c = std::make_shared<Buffer<T>>(M*N);             \
    broadcast_##op_name##_inner_raw(M, N,                         \
      buffer_a->data().data(),                                    \
      buffer_b->data().data(),                                    \
      buffer_c->data().data()                                     \
    );                                                            \
    return Tensor<T>(Shape::make(M, N), std::move(buffer_c));     \
  }

  BROADCAST_INNER(add);
  BROADCAST_INNER(sub);
  BROADCAST_INNER(mul);
  BROADCAST_INNER(div);

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
