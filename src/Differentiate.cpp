#include <libkann/Differentiate.hpp>

#include <libkann/Variable.hpp>

#include <libkann/Operation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <unordered_set>

#include <range/v3/all.hpp>

namespace kann
{
  namespace
  {
    struct TopologicalSortState
    {
      std::vector<variable_t> ordering;
      std::unordered_set<variable_t> visited;
    };

    static inline void topological_sort_impl(TopologicalSortState& state, variable_t variable)
    {
      if(state.visited.contains(variable))
        return;

      state.visited.insert(variable);
      for(const auto& input : variable->inputs)
        topological_sort_impl(state, input);

      state.ordering.push_back(variable);
    }

    static inline std::vector<variable_t> topological_sort(const std::vector<variable_t>& variables)
    {
      TopologicalSortState state;
      for(const auto& variable : variables)
        topological_sort_impl(state, variable);
      return state.ordering;
    }

    struct Info
    {
      std::vector<variable_t> gradients;
      variable_t gradient;
    };
  }

  std::unordered_map<variable_t, variable_t> differentiate(
    const std::vector<variable_t>& variables,
    const std::vector<variable_t>& gradients)
  {
    std::unordered_map<variable_t, Info> infos_map;
    ranges::actions::insert(infos_map, ranges::views::zip(variables, gradients | ranges::views::transform([](const auto& gradient){ return Info{.gradient = gradient}; })));

    auto ordering = topological_sort(variables);
    for(const auto& variable : ranges::views::reverse(ordering))
    {
      Info& info = infos_map.at(variable);
      if(!info.gradient)
      {
        assert(!info.gradients.empty());
        info.gradient = info.gradients.size() != 1
          ? Variable::apply(ReduceOperation(info.gradients.size()), info.gradients)
          : info.gradients.front();
      }

      // Do the differentiation
      if(variable->op)
      {
        auto gradients = variable->op->gradients(info.gradient, variable->inputs);
        for(const auto& [input, gradient] : ranges::views::zip(variable->inputs, gradients))
          infos_map[input].gradients.push_back(std::move(gradient));
      }
    }

    auto gradients_map = infos_map | ranges::views::transform([](const auto& p){
        const auto& [variable, info] = p;
        return std::make_pair(variable, info.gradient);
    });
    return ranges::to<std::unordered_map<variable_t, variable_t>>(gradients_map);
  }
}
