#include "App.hpp"

#include "Match.hpp"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cereal/archives/json.hpp>

AppGeneratePopulations::AppGeneratePopulations(const char* outputDirectory, seed_type seed) 
  : m_outputDirectory(outputDirectory), m_population(seed, 100) 
{
  std::clog << "Running with seed " << seed << '\n';
}

void AppGeneratePopulations::run()
{
  for(size_t i=0;;++i)
  {
    auto directoryPath = m_outputDirectory / (std::string("population")+std::to_string(i));
    m_population.writeTo(directoryPath);
    m_population.select();
  }
}

namespace
{
  Agent loadAgent(const char* filePath)
  {
    std::ifstream inputFile;
    inputFile.open(filePath);
    if(!inputFile)
      throw std::runtime_error(std::string("Failed to open file ") + filePath);

    cereal::JSONInputArchive inputArchive(inputFile);
    Agent agent;
    inputArchive(agent);
    return agent;
  }
}

AppMatchAgents::AppMatchAgents(const char* agent1FilePath, const char* agent2FilePath)
  : m_agent1(loadAgent(agent1FilePath)), m_agent2(loadAgent(agent2FilePath)) {}

void AppMatchAgents::run()
{
  auto result = match(m_agent1, m_agent2, 1000);
  std::cout << "End result of first game\n";
  std::cout << result.board1;
  std::cout << "End result of second game\n";
  std::cout << result.board2;
  switch(result.winningAgent)
  {
  case AgentID::NONE:
    std::cout << "Draw\n";
    break;
  case AgentID::_1:
    std::cout << "First agent won\n";
    break;
  case AgentID::_2:
    std::cout << "Second agent won\n";
    break;
  }
}

AppMatchAgentPlayer::AppMatchAgentPlayer(const char* agentFilePath)
  : m_agent(loadAgent(agentFilePath)) {}

void AppMatchAgentPlayer::run()
{
}
