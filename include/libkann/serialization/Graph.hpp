#pragma once

#include <cereal/cereal.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/graph_traits.hpp>

#include <map>
#include <type_traits>

template<typename Graph>
class GraphOutputSerializer
{
public:
  typedef boost::graph_traits<Graph> Traits;

public:
  GraphOutputSerializer(const Graph& graph) : m_graph(graph) {}

public:
  template<typename Archive>
  void save(Archive& archive) const
  {
    typename Traits::vertices_size_type verticesCount;
    typename Traits::edges_size_type edgesCount;

    verticesCount = boost::num_vertices(m_graph);
    edgesCount    = boost::num_edges(m_graph);

    archive(verticesCount);
    archive(edgesCount);

    size_t i = 0;
    for(auto [it, end] = boost::vertices(m_graph); it != end; ++it, ++i)
    {
      auto desc = *it;

      m_map.insert({desc, i});

      archive(boost::get(boost::vertex_all, m_graph, desc));
    }

    for(auto [it, end] = boost::edges(m_graph); it != end; ++it)
    {
      size_t u, v;
      typename Traits::vertex_descriptor source, target;

      auto desc = *it;

      source = boost::source(desc, m_graph);
      target = boost::target(desc, m_graph);

      u = m_map[source];
      v = m_map[target];

      archive(u, v);
      archive(boost::get(boost::edge_all, m_graph, desc));
    }
  }

  size_t map(typename Traits::vertex_descriptor desc) const { return m_map[desc]; }

private:
  const Graph& m_graph;
  mutable std::map<typename Traits::vertex_descriptor, size_t> m_map;
};

template<typename Graph>
class GraphInputSerializer
{
public:
  typedef boost::graph_traits<Graph> Traits;

public:
  GraphInputSerializer(Graph& graph) : m_graph(graph) {}

public:
  template<typename Archive>
  void load(Archive& archive)
  {
    typename Traits::vertices_size_type verticesCount;
    typename Traits::edges_size_type edgesCount;

    archive(verticesCount);
    archive(edgesCount);

    for(size_t i=0; i<verticesCount; ++i)
    {
      auto desc = boost::add_vertex(m_graph);

      m_map.push_back(desc);

      archive(boost::get(boost::vertex_all, m_graph, desc));
    }

    for(size_t i=0; i<edgesCount; ++i)
    {
      size_t u, v;
      typename Traits::vertex_descriptor source, target;

      archive(u, v);

      source = m_map[u];
      target = m_map[v];

      auto [desc, success] = boost::add_edge(source, target, m_graph);
      if(!success)
        throw std::runtime_error("Duplicate edges when deserializing boost graph");

      archive(boost::get(boost::edge_all, m_graph, desc));
    }
  }

  typename Traits::vertex_descriptor map(size_t i) { return m_map[i]; }

private:
  Graph& m_graph;
  std::vector<typename Traits::vertex_descriptor> m_map;
};

