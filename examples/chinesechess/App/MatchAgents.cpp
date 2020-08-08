#include "MatchAgents.hpp"

#include "../Match.hpp"

namespace App
{
  MatchAgents::MatchAgents(const char* agent1FilePath, const char* agent2FilePath)
    : m_agent1(Agent::loadFromFile(agent1FilePath)), m_agent2(Agent::loadFromFile(agent2FilePath)) {}

  void MatchAgents::run()
  {
    auto result = match(m_agent1, m_agent2, 1000);
    std::cout << "End result of first game\n";
    std::cout << result.board1;
    std::cout << "End result of second game\n";
    std::cout << result.board2;
    switch(result.winningAgent)
    {
    case PlayerID::NONE:
      std::cout << "Draw\n";
      break;
    case PlayerID::AGENT1:
      std::cout << "First agent won\n";
      break;
    case PlayerID::AGENT2:
      std::cout << "Second agent won\n";
      break;
    default:
      throw std::runtime_error("WTH has happend!");
    }
  }
}
