#include <libkann/layer_defs/Sequential.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <range/v3/all.hpp>

#include <span>

#include <assert.h>

namespace kann
{
  std::shared_ptr<Layer> SequentialLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->sub_layers = sub_layer_defs
      | ranges::views::transform([&prng](const auto& sub_layer_def) { return sub_layer_def->create(prng); })
      | ranges::to_vector;
    return layer;
  }

  size_t SequentialLayerDef::input_size() const
  {
    assert(!this->sub_layer_defs.empty());
    return this->sub_layer_defs.front()->input_size();
  }

  size_t SequentialLayerDef::output_size() const
  {
    assert(!this->sub_layer_defs.empty());
    return this->sub_layer_defs.back()->output_size();
  }

  template<typename T, typename R>
  static inline auto split_by(std::span<T> data, R&& sizes)
  {
    return ranges::views::transform(std::forward<R>(sizes) | ranges::views::exclusive_scan(0), std::forward<R>(sizes), [data](size_t index, size_t size){
        return data.subspan(index, size);
    });
  }

  LayerDef::ProcessOutput SequentialLayerDef::process(ProcessInput process_input) const
  {
    auto input  = process_input.variable;
    auto output = process_input.variable;

    auto parameters = std::move(process_input.parameters);
    auto states     = std::move(process_input.states);

    auto parameters_indices = sub_layer_defs | ranges::views::transform([](const auto& def) { return def->parameters_all_count();});
    auto states_indices     = sub_layer_defs | ranges::views::transform([](const auto& def) { return def->states_all_count();});

    auto parameters_subviews = split_by(std::span(parameters), parameters_indices);
    auto states_subviews     = split_by(std::span(states),     states_indices);

    for(auto [sub_layer_def, parameters_subview, states_subview] : ranges::views::zip(sub_layer_defs, parameters_subviews, states_subviews))
    {
      ProcessInput sub_process_input;
      sub_process_input.variable   = output;
      sub_process_input.parameters = parameters_subview | ranges::to_vector;
      sub_process_input.states     = states_subview     | ranges::to_vector;

      ProcessOutput sub_process_output = sub_layer_def->process(sub_process_input);
      output = sub_process_output.variable;
      ranges::move(sub_process_output.states, states_subview.begin());
    }

    return ProcessOutput{
      .variable = std::move(output),
      .states   = std::move(states)
    };
  }
}
