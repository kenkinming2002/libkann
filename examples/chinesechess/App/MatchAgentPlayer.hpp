#pragma once

#include <SFML/Graphics/RenderWindow.hpp>

#include "../Renderer.hpp"
#include "../Agent.hpp"
#include "../Human.hpp"

namespace App
{
  class MatchAgentPlayer
  {
  public:
    MatchAgentPlayer(const char* agentFilePath);

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
    bool m_matchEnded = false;

    Board m_board;
    Agent m_agent;
    Human m_human;
  };
}
