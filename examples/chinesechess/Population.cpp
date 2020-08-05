#include "Population.hpp"

Population::Population(seed_type seed, size_t size) : m_generator(seed)
{
  assert(size % 2 == 0 && "population size must be even");
  m_agents.reserve(size);
  std::generate_n(std::back_inserter(m_agents), size, [this](){
    auto topology = dynarray<size_t>{Agent::INPUT_LAYER_SIZE, 31, 31, 37, Agent::OUTPUT_LAYER_SIZE};
    auto neuralNetwork = NeuralNetwork(std::move(topology), m_generator, ActivationFunction(ActivationFunction::Type::SIGMOID));
    return Agent(std::move(neuralNetwork));
  });
}

void Population::select()
{
  constexpr static double MUTATION_RATE = 0.05;
  constexpr static size_t NUM_ITERATIONS = 2;

  const size_t size = m_agents.size() / 2;

  // Score
  for(size_t i=0; i<NUM_ITERATIONS; ++i)
  {
    std::shuffle(m_agents.begin(), m_agents.end(), m_generator);
    for(size_t j=0; j<m_agents.size(); j+=2)
    {
      auto& agent1 = m_agents[j+0];
      auto& agent2 = m_agents[j+1];
      auto result = Agent::match(agent1, agent2, 1000);
      std::cout << "Match " << (j/2)+1 << '\n';

      std::cout << result.board1;
      std::cout << "=========\n";

      std::cout << result.board2;
      std::cout << "=========\n";
    }
    std::cout << "Iteration " << i+1 << '\n';
  }

  // Eliminate
  std::sort(m_agents.begin(), m_agents.end(), [](const Agent& lhs, const Agent& rhs){
      return lhs.score() > rhs.score();
  });
  m_agents.erase(std::next(m_agents.begin(), size/2), m_agents.end());

  // Clear score
  std::for_each(m_agents.begin(), m_agents.end(), std::mem_fn(&Agent::clearScore));

  // Cross
  std::generate_n(std::back_inserter(m_agents), size/2, [&](){
      std::uniform_int_distribution<size_t> indexDistribution(0, size/2-1);
      const Agent& agent1 = m_agents[indexDistribution(m_generator)];
      const Agent& agent2 = m_agents[indexDistribution(m_generator)];
      return Agent::cross(agent1, agent2, m_generator, MUTATION_RATE);
  });

  assert(m_agents.size() == size);
}
