#pragma once

#include <cereal/cereal.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/graph_traits.hpp>

#include <map>
#include <type_traits>

namespace cereal
{
  template<typename Archive, class OS, class VS, class DS, class V, class E, class G, class ES>
  void save(Archive& archive, const boost::adjacency_list<OS, VS, DS, V, E, G, ES>& graph)
  {
    typedef boost::adjacency_list<OS, VS, DS, V, E, G, ES>              graph_type;
    typedef typename boost::graph_traits<graph_type>::vertex_descriptor vertex_type;
    typedef typename boost::graph_traits<graph_type>::edge_descriptor   edge_type;

    std::map<vertex_type, size_t> verticesMap;

    // Vertices
    size_t i = 0;
    size_t verticesCount = boost::num_vertices(graph);
    archive(verticesCount);
    for(auto [it, end] = boost::vertices(graph); it != end; ++it)
    {
      vertex_type vertex = *it;
      verticesMap.insert(std::make_pair(vertex, i++));
      archive(boost::get(boost::vertex_all, graph, vertex));
    }

    // Edges
    size_t edgesCount = boost::num_edges(graph);
    archive(edgesCount);
    for(auto [it, end] = boost::edges(graph); it != end; ++it)
    {
      edge_type edge = *it;

      vertex_type source = boost::source(edge, graph);
      vertex_type target = boost::target(edge, graph);

      size_t u = verticesMap.at(source);
      size_t v = verticesMap.at(target);

      archive(u, v);
      archive(boost::get(boost::edge_all, graph, edge));
    }
  }

  template<typename Archive, class OS, class VS, class DS, class V, class E, class G, class ES>
  void load(Archive& archive, boost::adjacency_list<OS, VS, DS, V, E, G, ES>& graph)
  {
    typedef boost::adjacency_list<OS, VS, DS, V, E, G, ES>              graph_type;
    typedef typename boost::graph_traits<graph_type>::vertex_descriptor vertex_type;

    std::vector<vertex_type> vertices;

    size_t verticesCount;
    archive(verticesCount);
    vertices.reserve(verticesCount);
    for(size_t i=0; i<verticesCount; ++i)
    {
      vertex_type vertex = boost::add_vertex(graph);
      vertices.push_back(vertex);
      archive(boost::get(boost::vertex_all, graph, vertex));
    }

    size_t edgesCount;
    archive(edgesCount);
    for(size_t i=0; i<edgesCount; ++i)
    {
      size_t u, v;
      archive(u, v);

      vertex_type source = vertices[u];
      vertex_type target = vertices[v];

      auto [edge, success] = boost::add_edge(source, target, graph);
      if(!success)
        throw std::runtime_error("Duplicate edges when deserializing boost graph");

      archive(boost::get(boost::edge_all, graph, edge));
    }
  }
}
