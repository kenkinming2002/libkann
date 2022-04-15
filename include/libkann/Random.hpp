#pragma once

#include <string.h>

#if defined(unix) || defined(__unix) || defined(__unix__)
#include <fstream>
#elif defined(_WIN32)
#endif

namespace kann
{
  namespace details
  {
    template <class To, class From>
    To bit_cast(const From& src) noexcept
      requires (sizeof(To) == sizeof(From)) && std::is_trivially_constructible_v<To>
    {
      To dst;
      memcpy(&dst, &src, sizeof(To));
      return dst;
    }
  }

#if defined(unix) || defined(__unix) || defined(__unix__)
  /*
   * The following uses /dev/urandom which is not guaranteed to exist. However,
   * most unix-like operating systems do support it.
   */
  template<typename Int>
  Int random()
  {
    char data[sizeof(Int)];

    std::ifstream random;
    /*
     * This disable internal buffering of ifstream to prevent depletion of entropy
     * of /dev/random. A more through explanation can be found via the
     * implementation of boost::random_device in the following link.
     *
     * https://github.com/boostorg/random/blob/develop/src/random_device.cpp
     */
    random.rdbuf()->pubsetbuf(nullptr, 0);
    random.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    random.open("/dev/urandom");

    random.read(data, sizeof data); // This may block if there is insufficient entropy

    return details::bit_cast<Int>(data);
  }

#elif defined(_WIN32)
    template<typename Int>
    Int random()
    {
      static_assert("Random number generation is not implemented on windows beacuse the author is unfamiliar with windows");
    }
#endif
}
