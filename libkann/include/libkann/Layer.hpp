#pragma once

#include <libkann/Function.hpp>
#include <libkann/Variable.hpp>

namespace kann
{
  struct LayerDef;
  struct KANN_EXPORT Layer : public Function
  {
  public:
    KANN_EXPORT void save_parameters(const std::string& dirname, bool include_gradient) const;
    KANN_EXPORT void load_parameters(const std::string& dirname, bool include_gradient);

  // Could I auto-magically generate them? Unfortunately, c++ have no reflection support.
  // All of them should be pretty easy to generate
  public:
    KANN_EXPORT virtual const LayerDef& get_def() const = 0;

    // Forwarded from LayerDef
    KANN_EXPORT virtual tensor::Shape get_input_shape()  const = 0;
    KANN_EXPORT virtual tensor::Shape get_output_shape() const = 0;

  public:
    KANN_EXPORT virtual void initialize(std::default_random_engine& prng) = 0;

    KANN_EXPORT virtual std::unordered_map<std::string, const Variable*> parameters_map() const = 0;
    KANN_EXPORT virtual std::unordered_map<std::string, Variable*>       parameters_map()       = 0;
  };

}
