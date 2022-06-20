#pragma once

#include <libkann/Types.hpp>
#include <libkann/Shape.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Tag.hpp>
#include <libkann/Graph.hpp>

namespace kann
{
  struct Context
  {
    Graph graph;

    // Config
    size_t batch_size;

    // Input/Output
    size_t inputs_index;
    size_t outputs_index;
    size_t expected_outputs_index;

    // Parameters
    std::vector<Shape> parameter_shapes;
    std::vector<Tag>    parameter_tags;
    std::vector<size_t> parameter_indices;

    std::vector<size_t> parameter_gradient_indices;

    std::vector<size_t> new_parameter_indices;

    // States
    std::vector<Shape> state_shapes;
    std::vector<size_t> input_state_indices;
    std::vector<size_t> output_state_indices;

    std::vector<Tensor> state_gradient_values;
    std::vector<size_t> state_gradient_indices;

    // Optimizer states
    std::vector<Tensor> optimizer_initial_state_values;
    std::vector<size_t> optimizer_input_state_indices;
    std::vector<size_t> optimizer_output_state_indices;

    void forward_pass(LayerStorage& layer, size_t batch_size);

    // TODO: Embed output size into graph vertex
    // TODO: Take in a loss function
    void gradient_pass(Shape output_shape);
    void backward_pass();
    void training_pass(const Optimizer& optimizer);
  };
}
