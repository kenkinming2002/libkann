#pragma once

#include <cereal/cereal.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/graph_traits.hpp>

#include <map>
#include <type_traits>

template<typename Graph>
class VertexSerializer;

template<typename Graph>
class GraphSerializer
{
public:
  typedef boost::graph_traits<Graph> Traits;

public:
  GraphSerializer(Graph& graph) : m_graph(graph) {}

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

      m_map1.insert({desc, i});
      m_map2.push_back(desc);

      archive(boost::get(boost::vertex_all, m_graph, desc));
    }

    for(auto [it, end] = boost::edges(m_graph); it != end; ++it)
    {
      size_t u, v;
      typename Traits::vertex_descriptor source, target;

      auto desc = *it;

      source = boost::source(desc, m_graph);
      target = boost::target(desc, m_graph);

      u = m_map1[source];
      v = m_map1[target];

      archive(u, v);
      archive(boost::get(boost::edge_all, m_graph, desc));
    }
  }

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

      m_map1.insert({desc, i});
      m_map2.push_back(desc);

      archive(boost::get(boost::vertex_all, m_graph, desc));
    }

    for(size_t i=0; i<edgesCount; ++i)
    {
      size_t u, v;
      typename Traits::vertex_descriptor source, target;

      archive(u, v);

      source = m_map2[u];
      target = m_map2[v];

      auto [desc, success] = boost::add_edge(source, target, m_graph);
      if(!success)
        throw std::runtime_error("Duplicate edges when deserializing boost graph");

      archive(boost::get(boost::edge_all, m_graph, desc));
    }
  }

private:
  Graph& m_graph;

private:
  mutable std::map<typename Traits::vertex_descriptor, size_t> m_map1;
  mutable std::vector<typename Traits::vertex_descriptor> m_map2;

private:
  friend class VertexSerializer<Graph>;
};

template<typename Graph>
class VertexSerializer
{
public:
  typedef boost::graph_traits<Graph> Traits;
  typedef std::conditional_t<std::is_const_v<Graph>, const typename Traits::vertex_descriptor, typename Traits::vertex_descriptor> VertexDescriptor;

public:
  VertexSerializer(const GraphSerializer<Graph>& graphSerializer, VertexDescriptor& desc)
    : m_graphSerializer(graphSerializer), m_desc(desc) {}

public:
  template<typename Archive>
  void save(Archive& archive) const
  {
    size_t i;

    i = m_graphSerializer.m_map1[m_desc];
    archive(i);
  }

  template<typename Archive>
  void load(Archive& archive)
  {
    size_t i;

    archive(i);
    m_desc = m_graphSerializer.m_map2[i];
  }

private:
  const GraphSerializer<Graph>& m_graphSerializer;
  VertexDescriptor& m_desc;
};

