#pragma once

#include <libkann/Export.hpp>

#include <libkann/Types.hpp>

#include <unordered_map>
#include <vector>
#include <optional>
#include <iosfwd>

#include <assert.h>

namespace kann
{
  /* A computational graph */
  class Graph
  {
  public:
    struct Vertex
    {
      std::optional<size_t> in_edge_index;
      std::vector<size_t> out_edge_indices;
      std::optional<size_t> gradient_index;
    };

    struct Edge
    {
      operation_t op;
      std::vector<size_t> input_indices;
      std::vector<size_t> output_indices;
    };

  public:
    size_t edges_count()    const { return m_edges.size(); }
    size_t vertices_count() const { return m_vertices.size(); }

  public:
    const Edge& edge(size_t index)     const { assert(index < edges_count());    return m_edges[index]; }
    const Vertex& vertex(size_t index) const { assert(index < vertices_count()); return m_vertices[index]; }

  public:
    KANN_EXPORT size_t add_vertex();
    KANN_EXPORT std::vector<size_t> add_vertices(size_t count);
    KANN_EXPORT void add_edge(operation_t op, std::vector<size_t> input_indices, std::vector<size_t> output_indices);

  public:
    KANN_EXPORT void set_gradient_index(size_t index, size_t gradient_index);
    KANN_EXPORT size_t get_gradient_index(size_t index);

  private:
    void differentiate(size_t edge_index);

  public:
    KANN_EXPORT void write_graphviz(std::ostream& os);

  private:
    std::vector<Vertex> m_vertices;
    std::vector<Edge> m_edges;
  };
}
