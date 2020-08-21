#pragma once

#include <SFML/Graphics/RenderWindow.hpp>

#include "../../Game.hpp"
#include "../../Renderer.hpp"
#include "../../Agent.hpp"

#include "../Match.hpp"

namespace App
{
  class Match::Agents
  {
  public:
    Agents(const char* agent1FileName, const char* agent2FileName);

  public:
    void run();

  public:
    bool loop();
    void handleInput();
    void update();
    void render();

  private:
    sf::RenderWindow m_window;
    Renderer m_renderer;

  private:
    Game m_game;
    Agent m_agent1, m_agent2;

  private:
    enum class AgentID
    {
      _1,
      _2
    } m_currentAgent = AgentID::_1;
  };

}
