#pragma once

#include <libkann/Export.hpp>
#include <libkann/Variable.hpp>

#include <vector>

namespace kann
{
  struct LayerStorage
  {
  public:
    std::vector<Variable> parameters;
    std::vector<Variable> states;

  public:
    std::vector<std::shared_ptr<LayerStorage>> sub_layer_storages;

  public:
    template<typename Callback>
    void foreach_parameters(Callback cb) requires(std::is_invocable_v<Callback, Variable&>)
    {
      for(Variable& parameter : parameters)
        cb(parameter);

      for(std::shared_ptr<LayerStorage>& sub_layer_storage : sub_layer_storages)
        sub_layer_storage->foreach_parameters(cb);
    }

    template<typename Callback>
    void foreach_states(Callback cb) requires(std::is_invocable_v<Callback, Variable&>)
    {
      for(Variable& state : states)
        cb(state);

      for(std::shared_ptr<LayerStorage>& sub_layer_storage : sub_layer_storages)
        sub_layer_storage->foreach_states(cb);
    }
  };
}
