#include <libtensor/Stack.hpp>

namespace tensor
{
  void stack_outer_raw(size_t M, size_t N, char* __restrict__ output, const char* __restrict__ * __restrict__ inputs, size_t size) noexcept
  {
    for(size_t m=0; m<M; ++m)
      std::copy_n(&inputs[m][0], N*size, &output[m*N*size]);
  }

  void stack_inner_raw(size_t M, size_t N, char* __restrict__ output, const char* __restrict__ * __restrict__ inputs, size_t size) noexcept
  {
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        std::copy_n(&inputs[m][n*size], size, &output[(n*M+m)*size]);
  }
}


