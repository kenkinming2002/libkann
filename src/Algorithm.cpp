#include <libkann/Algorithm.hpp>

#include <libkann/LayerDef.hpp>

#include <libkann/Differentiate.hpp>
#include <libkann/operations/MultiplyOperation.hpp>
#include <libkann/operations/SubtractOperation.hpp>

#include <range/v3/all.hpp>
#include <fmt/core.h>

#include <numeric>
#include <algorithm>

namespace kann
{
  namespace helpers
  {
    template<typename T>
    static inline void move_append(std::vector<T>& target, std::vector<T> from)
    {
      target.insert(target.end(),
        std::move_iterator(from.begin()),
        std::move_iterator(from.end())
      );
    }

    template<typename T, size_t N>
    static inline std::vector<T> move_join(std::array<std::vector<T>, N> data)
    {
      std::vector<T> result;

      size_t size = std::accumulate(data.begin(), data.end(), 0, [](size_t size, const std::vector<T>& data){
          return size + data.size();
      });
      result.reserve(size);

      std::for_each(data.begin(), data.end(), [&result](std::vector<T>& data){
          result.insert(result.end(),
              std::move_iterator(data.begin()),
              std::move_iterator(data.end())
          );
      });

      return result;
    }

    template<typename... Args>
    static inline auto move_join(Args&&... args) { return move_join(std::array{std::forward<Args>(args)...}); }

    static inline std::vector<std::shared_ptr<const Variable>> create_input_variables(size_t count)
    {
      std::vector<std::shared_ptr<const Variable>> result;
      result.reserve(count);
      for(size_t i=0; i<count; ++i)
        result.push_back(std::make_shared<const Variable>()); // TODO: Add distinct name to each variable

      return result;
    }

    template<typename T, typename Func, typename Ret = std::invoke_result_t<Func, const T&>>
    static inline std::vector<Ret> map1(const std::vector<T>& variables, Func func)
    {
      size_t size = variables.size();

      std::vector<std::shared_ptr<const Variable>> result;
      result.reserve(size);
      for(size_t i=0; i<size; ++i)
        result.push_back(func(variables[i]));

      return result;
    }

    template<typename T, typename U, typename Func, typename Ret = std::invoke_result_t<Func, const T&, const U&>>
    static inline std::vector<Ret> map2(const std::vector<T>& variables1, const std::vector<U>& variables2, Func func)
    {
      assert(variables1.size() == variables2.size());
      size_t size = variables1.size();

      std::vector<std::shared_ptr<const Variable>> result;
      result.reserve(size);
      for(size_t i=0; i<size; ++i)
        result.push_back(func(variables1[i], variables2[i]));

      return result;
    }
  }

}
