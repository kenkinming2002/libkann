#include <boost/graph/topological_sort.hpp>
#include <libkann/Differentiate.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

namespace kann
{
  namespace
  {
    struct Node
    {
      std::shared_ptr<const Variable> variable;

      std::vector<std::shared_ptr<const Variable>> gradients;
      std::shared_ptr<const Variable>              gradient;
    };

    struct Connection
    {
      size_t id;
    };

    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      Node, Connection
    > graph_type;
    typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_type;

  }

  template<typename Callback>
  static void walk(const std::shared_ptr<const Variable>& variable, const Callback& callback)
  {
    if(!callback(variable))
      return;

    for(const auto& input : variable->inputs)
      walk(input, callback);
  }

  std::unordered_map<std::shared_ptr<const Variable>, std::shared_ptr<const Variable>> differentiate(
    const std::vector<std::shared_ptr<const Variable>>& variables,
    const std::vector<std::shared_ptr<const Variable>>& gradients)
  {
    graph_type graph;

    // 1: Build the graph
    std::unordered_map<std::shared_ptr<const Variable>, vertex_type> verticesMap;
    for(const auto& variable : variables)
      walk(variable, [&](const auto& variable){
        auto it = verticesMap.find(variable);
        if(it != verticesMap.end())
          return false;

        auto vertex = boost::add_vertex(Node{.variable = variable}, graph);
        verticesMap.emplace(variable, vertex);
        return true;
      });

    for(const auto& [variable, vertex] : verticesMap)
      for(size_t i=0; i<variable->inputs.size(); ++i)
      {
        const auto& inputVariable = variable->inputs[i];
        const auto& inputVertex = verticesMap.at(inputVariable);
        boost::add_edge(inputVertex, vertex, Connection{.id = i}, graph);
      }

    // 2: Associate the gradients
    {
      assert(variables.size() == gradients.size());
      const size_t size = variables.size();
      for(size_t i=0; i<size; ++i)
      {
        const vertex_type vertex = verticesMap.at(variables[i]);
        graph[vertex].gradients.push_back(gradients[i]);
      }
    }

    // 3: Evaluate the gradients recursively
    std::vector<vertex_type> ordering;
    boost::topological_sort(graph, std::back_inserter(ordering));

    for(const vertex_type vertex : ordering)
    {
      Node& node = graph[vertex];

      assert(node.gradients.size() != 0);
      if(node.gradients.size() == 1)
        node.gradient = std::make_shared<const Variable>(node.gradients, std::make_shared<IdentityOperation>());
      else
        node.gradient = std::make_shared<const Variable>(node.gradients, std::make_shared<ReduceOperation>(node.gradients.size()));

      if(node.variable->op)
      {
        const auto& inputs = node.variable->inputs;
        const auto& gradients = node.variable->op->gradients(node.gradient, inputs);
        for(auto [it, end] = boost::in_edges(vertex, graph); it != end; ++it)
        {
          const Connection& connection = graph[*it];
          Node& inputNode = graph[boost::source(*it, graph)];
          inputNode.gradients.push_back(gradients[connection.id]);
        }
      }
    }

    // 4: Create the resulting map
    std::unordered_map<std::shared_ptr<const Variable>, std::shared_ptr<const Variable>> map;
    for(auto [it, end] = boost::vertices(graph); it != end; ++it)
    {
      Node& node = graph[*it];
      map.emplace(std::move(node.variable), std::move(node.gradient));
    }
    return map;
  }
}
