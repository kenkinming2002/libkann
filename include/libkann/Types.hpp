#pragma once

#include <libkann/Tensor.hpp>
#include <libkann/Variable.hpp>
#include <libkann/QualifiedName.hpp>

#include <memory>
#include <unordered_map>

namespace kann
{
  typedef std::shared_ptr<const Tensor> TRef;
  typedef std::shared_ptr<const Variable> VRef;

  typedef std::unordered_map<QualifiedName, TRef> TMap;
  typedef std::unordered_map<QualifiedName, VRef> VMap;
}
