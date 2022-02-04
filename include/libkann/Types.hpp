#pragma once

#include <libkann/Tensor.hpp>
#include <libkann/Variable.hpp>
#include <libkann/Parameter.hpp>

#include <memory>
#include <unordered_map>

namespace kann
{
  typedef std::shared_ptr<const Tensor> TRef;
  typedef std::shared_ptr<const Variable> VRef;

  typedef std::unordered_map<Parameter, TRef> TMap;
  typedef std::unordered_map<Parameter, VRef> VMap;
}
