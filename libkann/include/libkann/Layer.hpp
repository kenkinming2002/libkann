#pragma once

#include <libkann/Function.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/Variable.hpp>

namespace kann
{
  struct KANN_EXPORT Layer : public Function
  {
  // Could I auto-magically generate them? Unfortunately, c++ have no reflection support.
  // All of them should be pretty easy to generate
  public:
    KANN_EXPORT virtual const LayerDef& get_def() const { assert(false && "Unimplemented"); }

    // Forwarded from LayerDef
    KANN_EXPORT virtual tensor::Shape get_input_shape()  const { assert(false && "Unimplemented"); }
    KANN_EXPORT virtual tensor::Shape get_output_shape() const { assert(false && "Unimplemented"); }

  public:
    KANN_EXPORT virtual void initialize(std::default_random_engine& prng) { assert(false && "Unimplemented"); }

    KANN_EXPORT virtual std::unordered_map<std::string, const Variable*> parameters_map() const { assert(false && "Unimplemented"); };
    KANN_EXPORT virtual std::unordered_map<std::string, Variable*>       parameters_map()       { assert(false && "Unimplemented"); };
  };
}
