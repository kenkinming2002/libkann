#include "Population.hpp"

#include "Match.hpp"

#include <cereal/archives/binary.hpp>

#include <cereal/details/helpers.hpp>
#include <fstream>
#include <string>

Population::Population(seed_type seed, size_t size, const std::vector<size_t>& hiddenLayers) : m_generator(seed)
{
  dynarray<size_t> topology(hiddenLayers.size() + 2);

  topology.front() = Agent::INPUT_LAYER_SIZE;
  std::copy(hiddenLayers.begin(), hiddenLayers.end(), std::next(topology.begin()));
  topology.back() = Agent::OUTPUT_LAYER_SIZE;

  assert(size % 2 == 0 && "population size must be even");
  m_agents.reserve(size);
  std::generate_n(std::back_inserter(m_agents), size, [&](){
    auto neuralNetwork = NeuralNetwork(topology, m_generator, ActivationFunction(ActivationFunction::Type::SIGMOID));
    return Agent(std::move(neuralNetwork));
  });
}

void Population::select(size_t iterations, double mutationRate)
{
  const size_t size = m_agents.size();

  // Score
  std::clog << "Scoring individual agents..." << std::flush;
  {
    for(size_t i=0; i<iterations; ++i)
    {
      std::shuffle(m_agents.begin(), m_agents.end(), m_generator);

#pragma omp parallel for
      for(size_t j=0; j<m_agents.size(); j+=2)
      {
        auto& agent1 = m_agents[j+0];
        auto& agent2 = m_agents[j+1];
        auto result = match(agent1, agent2, 1000);
        agent1.addScore(result.score1);
        agent2.addScore(result.score2);
      }
    }
  }
  std::clog << "Done\n";

  // Eliminate
  std::clog << "Eliminating incompetent agents..." << std::flush;
  {
    std::sort(m_agents.begin(), m_agents.end(), [](const Agent& lhs, const Agent& rhs){
        return lhs.score() > rhs.score();
    });
    m_agents.erase(std::next(m_agents.begin(), size/2), m_agents.end());

    // Clear score
    std::for_each(m_agents.begin(), m_agents.end(), std::mem_fn(&Agent::clearScore));
  }
  std::clog << "Done\n";

  // Cross
  std::clog << "Generating new agents from survivor..." << std::flush;
  {
    std::generate_n(std::back_inserter(m_agents), size/2, [&](){
        std::uniform_int_distribution<size_t> indexDistribution(0, size/2-1);
        const Agent& agent1 = m_agents[indexDistribution(m_generator)];
        const Agent& agent2 = m_agents[indexDistribution(m_generator)];
        return Agent::cross(agent1, agent2, m_generator, mutationRate);
    });
  }
  std::clog << "Done\n";
}


void Population::writeTo(const std::filesystem::path& directory) const
{
  std::filesystem::create_directories(directory);
  for(size_t i=0; i<m_agents.size(); ++i)
  {
    const auto& agent = m_agents[i];
    agent.saveToFile((directory / ("agent"+std::to_string(i))).c_str());
  }
}
