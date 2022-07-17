#include <libtensor/memops/Split.hpp>

namespace tensor
{
  void split_outer_raw(size_t M, size_t N, char* __restrict__ * __restrict__ outputs, const char* __restrict__ input, size_t size) noexcept
  {
    for(size_t m=0; m<M; ++m)
      std::copy_n(&input[m*N*size], N*size, &outputs[m][0]);
  }

  void split_inner_raw(size_t M, size_t N, char* __restrict__ * __restrict__ outputs, const char* __restrict__ input, size_t size) noexcept
  {
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        std::copy_n(&input[(n*M+m)*size], size, &outputs[m][n*size]);
  }
}
