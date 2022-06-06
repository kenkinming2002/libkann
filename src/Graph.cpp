#include <libkann/Graph.hpp>

#include <libkann/Operation.hpp>

#include <range/v3/all.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <cxxabi.h>

namespace kann
{
  size_t Graph::add_vertex()
  {
    size_t vertex_index = m_vertices.size();
    m_vertices.push_back(Vertex{});
    return vertex_index;
  }

  void Graph::add_edge(operation_t op, std::vector<size_t> input_indices, std::vector<size_t> output_indices)
  {
    size_t edge_index = m_edges.size();

    for(size_t input_index : input_indices)
      m_vertices[input_index].out_edge_indices.push_back(edge_index);

    for(size_t output_index : output_indices)
    {
      assert(!m_vertices[output_index].in_edge_index);
      m_vertices[output_index].in_edge_index = edge_index;
    }

    m_edges.push_back(Edge{
      .op             = std::move(op),
      .input_indices  = std::move(input_indices),
      .output_indices = std::move(output_indices)
    });
  }

  static inline std::string demangle(const char* name)
  {
    int status;
    char* demangled_name = abi::__cxa_demangle(name, 0, 0, &status);
    if(!demangled_name)
      return name;

    std::string tmp = demangled_name;
    free(demangled_name);
    return tmp;
  }

  void Graph::set_gradient_index(size_t index, size_t gradient_index)
  {
    assert(!m_vertices[index].gradient_index);
    m_vertices[index].gradient_index = gradient_index;
  }

  size_t Graph::get_gradient_index(size_t index)
  {
    if(!m_vertices[index].gradient_index)
    {
      // Ensure it has only one out-going edges
      assert(m_vertices[index].out_edge_indices.size() == 1);
      size_t edge_index = m_vertices[index].out_edge_indices.front();
      differentiate(edge_index);
    }

    assert(m_vertices[index].gradient_index);
    return *m_vertices[index].gradient_index;
  }

  void Graph::differentiate(size_t edge_index)
  {
    // Copy to avoid invalidation
    std::vector<size_t> input_indices  = m_edges[edge_index].input_indices;
    std::vector<size_t> output_indices = m_edges[edge_index].output_indices;
    operation_t gradient_op            = m_edges[edge_index].op->differentiate();

    std::vector<size_t> input_gradient_indices  = input_indices | ranges::views::transform([this](size_t input_index)
    {
      size_t input_gradient_index = add_vertex();
      set_gradient_index(input_index, input_gradient_index);
      return input_gradient_index;
    }) | ranges::to_vector;

    std::vector<size_t> output_gradient_indices = output_indices | ranges::views::transform([this](size_t output_index)
    {
      return get_gradient_index(output_index);
    }) | ranges::to_vector;

    add_edge(gradient_op,
      ranges::views::concat(input_indices, output_gradient_indices) | ranges::to_vector,
      input_gradient_indices
    );
  }

  void Graph::write_graphviz(std::ostream& os)
  {
    fmt::print(os, "digraph {{\n");

    // Vertices
    for(const auto& [i, vertex] : ranges::views::enumerate(m_vertices))
    {
      const char* color = [vertex=vertex](){
        if(vertex.in_edge_index && !vertex.out_edge_indices.empty())
          return "black";

        if(!vertex.out_edge_indices.empty())
          return "red";

        if(vertex.in_edge_index)
          return "green";

        return "gray";
      }();
      fmt::print(os, "vertex_{} [color={}];\n", i, color);
    }

    // Edges
    for(const auto& [i, edge] : ranges::views::enumerate(m_edges))
    {
      const Operation& op = *edge.op;
      const char* op_name = typeid(op).name();
      fmt::print(os, "edge_{} [color=transparent,label=\"{}\"];\n", i, demangle(op_name).substr(6));

      for(size_t input_index : edge.input_indices)
        fmt::print(os, "vertex_{} -> edge_{};\n", input_index, i);

      for(size_t output_index : edge.output_indices)
        fmt::print(os, "edge_{} -> vertex_{};\n", i, output_index);
    }

    fmt::print(os, "}}\n");
  }
}
