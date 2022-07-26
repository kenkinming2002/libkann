#pragma once

#include <libkann/Function.hpp>
#include <libkann/Variable.hpp>

namespace kann
{
  struct LayerDef;
  struct LIBKANN_EXPORT Layer : public Function
  {
  // Could I auto-magically generate them? Unfortunately, c++ have no reflection support.
  // All of them should be pretty easy to generate
  public:
    LIBKANN_EXPORT virtual const LayerDef& get_def() const = 0;

  public:
    LIBKANN_EXPORT virtual void initialize(std::default_random_engine& prng) = 0;

    LIBKANN_EXPORT virtual std::unordered_map<std::string, const Variable*> parameters_map() const = 0;
    LIBKANN_EXPORT virtual std::unordered_map<std::string, Variable*>       parameters_map()       = 0;
  };

}
