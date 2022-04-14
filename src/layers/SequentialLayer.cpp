#include <libkann/layers/SequentialLayer.hpp>

#include <span>
#include <assert.h>

namespace kann
{
  template<typename T>
  static std::span<T> split_front(std::span<T>& data, size_t count)
  {
    assert(data.size() >= count);
    auto result = data.first(count);
    data = data.last(data.size() - count);
    return result;
  }

  template<typename T>
  static std::span<T> split_back(std::span<T>& data, size_t count)
  {
    assert(data.size() >= count);
    auto result = data.last(count);
    data = data.first(data.size() - count);
    return result;
  }

  template<typename T>
  static std::vector<T> move_to_vector(std::span<T> source)
  {
    return std::vector(std::move_iterator(source.begin()), std::move_iterator(source.end()));
  }

  template<typename T>
  static void move_from_vector(std::span<T> target, std::vector<T>&& data)
  {
    std::move(std::move_iterator(data.begin()), std::move_iterator(data.end()), target.begin());
  }

  SequentialLayer::SequentialLayer(const SequentialLayer& other)
    : m_taggedLayers(other.m_taggedLayers)
  {
    for(auto& [tag, layer] : m_taggedLayers)
      layer = layer->clone();
  }

  void SequentialLayer::addLayer(std::shared_ptr<Layer> layer, Tag tag)
  {
    m_taggedLayers.push_back(TaggedLayer{
      .tag   = tag,
      .layer = std::move(layer)
    });
  }

  std::shared_ptr<Layer> SequentialLayer::clone() const
  {
    return std::make_shared<SequentialLayer>(*this);
  }

  void SequentialLayer::randomize(std::default_random_engine& engine)
  {
    for(auto& [tag, layer] : m_taggedLayers)
      layer->randomize(engine);
  }

  size_t SequentialLayer::input_size() const
  {
    assert(!m_taggedLayers.empty());
    return m_taggedLayers.front().layer->input_size();
  }

  size_t SequentialLayer::output_size() const
  {
    assert(!m_taggedLayers.empty());
    return m_taggedLayers.back().layer->output_size();
  }

  size_t SequentialLayer::parameters_count() const
  {
    size_t count = 0;
    for(const auto& [tag, layer] : m_taggedLayers)
      count += layer->parameters_count();

    return count;
  }

  size_t SequentialLayer::states_count() const
  {
    size_t count = 0;
    for(const auto& [tag, layer] : m_taggedLayers)
      count += layer->states_count();

    return count;
  }

  std::vector<size_t> SequentialLayer::parameter_sizes() const
  {
    std::vector<size_t> sizes;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      const auto tmp = layer->parameter_sizes();
      sizes.insert(sizes.end(), tmp.begin(), tmp.end());
    }
    return sizes;
  }

  std::vector<size_t> SequentialLayer::state_sizes() const
  {
    std::vector<size_t> sizes;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      const auto tmp = layer->state_sizes();
      sizes.insert(sizes.end(), tmp.begin(), tmp.end());
    }
    return sizes;
  }

  std::vector<std::shared_ptr<const Tensor>> SequentialLayer::get_parameters() const
  {
    std::vector<std::shared_ptr<const Tensor>> values;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      const auto tmp = layer->get_parameters();
      values.insert(values.end(), tmp.begin(), tmp.end());
    }
    return values;
  }

  std::vector<std::shared_ptr<const Tensor>> SequentialLayer::get_states() const
  {
    std::vector<std::shared_ptr<const Tensor>> values;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      const auto tmp = layer->get_states();
      values.insert(values.end(), tmp.begin(), tmp.end());
    }
    return values;
  }

  void SequentialLayer::set_parameters(std::vector<std::shared_ptr<const Tensor>> values)
  {
    auto view = std::span(values);
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto subview = split_front(view, layer->parameters_count());
      auto tmp = std::vector(std::move_iterator(subview.begin()), std::move_iterator(subview.end()));
      layer->set_parameters(std::move(tmp));
    }
    assert(view.empty());
  }

  void SequentialLayer::set_states(std::vector<std::shared_ptr<const Tensor>> values)
  {
    auto view = std::span(values);
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto subview = split_front(view, layer->states_count());
      auto tmp = std::vector(std::move_iterator(subview.begin()), std::move_iterator(subview.end()));
      layer->set_states(std::move(tmp));
    }
    assert(view.empty());
  }

  Layer::ProcessOutput SequentialLayer::process(ProcessInput input) const
  {
    auto parameters = std::move(input.parameters);
    auto states     = std::move(input.states);

    auto parameters_view = std::span(parameters);
    auto states_view     = std::span(states);

    ProcessOutput output;
    for(const auto& [tag, layer] : m_taggedLayers)
    {
      auto parameters_subview = split_front(parameters_view, layer->parameters_count());
      auto states_subview     = split_front(states_view,     layer->states_count());

      input.parameters = move_to_vector(parameters_subview);
      input.states     = move_to_vector(states_subview);

      output = layer->process(std::move(input));
      input.variable = output.variable;

      move_from_vector(states_subview, std::move(output.states));
    }

    assert(parameters_view.empty());
    assert(states_view.empty());

    output.states = std::move(states);
    return output;
  }
}
