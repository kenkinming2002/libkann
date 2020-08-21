#pragma once

#include <SFML/Graphics/RenderWindow.hpp>

#include "../../Game.hpp"
#include "../../Renderer.hpp"
#include "../../Agent.hpp"
#include "../../Human.hpp"

#include "../Match.hpp"

namespace App
{
  class Match::AgentPlayer
  {
  public:
    AgentPlayer(const char* agentFileName);

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
    Agent m_agent;
    Human m_human;
  };
}
