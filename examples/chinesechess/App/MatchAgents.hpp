#pragma once

#include "../Agent.hpp"

namespace App
{
  class MatchAgents
  {
  public:
    MatchAgents(const char* agent1FilePath, const char* agent2FilePath);

  public:
    void run();

  private:
    Agent m_agent1, m_agent2;
  };

}
