#include "Population.hpp"

#include "Game.hpp"

#include <libkann/layers/SequentialLayer.hpp>
#include <libkann/layers/WeightLayer.hpp>
#include <libkann/layers/ActivationLayer.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/details/helpers.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

Population::Population(std::default_random_engine& engine, size_t size)
{
  m_agents.reserve(size);
  for(size_t i=0; i<size; ++i)
    m_agents.push_back(AIAgent::make(engine));
}

void Population::select(size_t iterations, std::default_random_engine& engine, double mutationRate)
{
  // Select and eliminate
  std::vector<std::pair<size_t, double>> scores;
  scores.reserve(m_agents.size());
  for(size_t i=0; i<m_agents.size(); ++i)
    scores.emplace_back(i, 0.0);

  std::cout << "Scoring..." << std::flush;
  for(size_t i=0; i<iterations; ++i)
  {
    std::shuffle(scores.begin(), scores.end(), engine);
    for(size_t j=0; j<m_agents.size(); j+=2)
    {
      auto& [index1, score1] = scores[j];
      auto& [index2, score2] = scores[j+1];

      AIAgent& agent1 = m_agents[index1];
      AIAgent& agent2 = m_agents[index2];

      GameResult result1 = game(agent1, agent2);
      GameResult result2 = game(agent2, agent1);

      score1 += result1.score1 + result2.score2;
      score2 += result1.score2 + result2.score1;
    }
  }
  std::cout << "Done\n";

  std::sort(scores.begin(), scores.end(), [](const auto& lhs, const auto& rhs) {
    const auto& [index1, score1] = lhs;
    const auto& [index2, score2] = rhs;
    return score1 < score2;
  });

  const size_t eliminateCount = scores.size()/2;

  std::cout << "Eliminating..." << std::flush;
  {
    std::vector<AIAgent> newAgents;
    for(size_t i=eliminateCount; i<scores.size(); ++i)
    {
      const auto& [index, score] = scores[i];
      newAgents.push_back(std::move(m_agents[index]));
    }
    m_agents = std::move(newAgents);
  }
  std::cout << "Done\n";

  std::cout << "Generating..." << std::flush;
  std::uniform_int_distribution<size_t> dist(0, m_agents.size()-1);
  for(size_t i=0; i<eliminateCount; ++i)
  {
    size_t index1, index2;
    do
    {
      index1 = dist(engine);
      index2 = dist(engine);
    }
    while(index1 == index2);

    auto agent = AIAgent::cross(m_agents[index1], m_agents[index2], engine, mutationRate);
    m_agents.push_back(std::move(agent));
  }
  std::cout << "Done\n";
}

void Population::write(const std::filesystem::path& path)
{
  std::filesystem::create_directories(path);
  for(size_t i=0; i<m_agents.size(); ++i)
  {
    std::stringstream ss;
    ss << "agent" << std::setfill('0') << std::setw(5) << i;
    auto filename = ss.str();

    std::ofstream file(path / filename);
    cereal::BinaryOutputArchive archive(file);

    archive(m_agents[i]);
  }
}
