#include "MatchAgentPlayer.hpp"

#include <SFML/Window/Event.hpp>
#include <SFML/Window/Keyboard.hpp>

static constexpr unsigned WINDOW_WIDTH = 1000;
static constexpr unsigned WINDOW_HEIGHT = 1000;

namespace App
{
  MatchAgentPlayer::MatchAgentPlayer(const char* agentFileName) 
    : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), ""),
      m_renderer(m_window)
  {
    m_agent.loadFromFile(agentFileName);
  }

  void MatchAgentPlayer::run()
  {
    while(this->loop());
  }

  bool MatchAgentPlayer::loop()
  {
    this->handleInput();
    this->update();
    this->render();
    return m_window.isOpen();
  }

  void MatchAgentPlayer::handleInput()
  {
    sf::Event event;
    while (m_window.pollEvent(event))
    {
      if(m_human.handleInput(m_renderer, m_game.board(), Board::Cell::Color::RED, event))
        continue;

      if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::U)
      {
        m_game.undoMove();
        m_game.undoMove();
      }

      if(event.type == sf::Event::Closed)
        m_window.close();
    }
  }

  void MatchAgentPlayer::update()
  {
    // Check
    if(m_game.ended())
      return;

    // Human
    if(auto move = m_human.pollMove(m_game.board(), Board::Cell::Color::RED))
      m_game.performMove(*move, Board::Cell::Color::RED);
    else
      return;

    // Check
    if(m_game.ended())
      return;

    // Agent
    if(auto move = m_agent.selectMove(m_game.board(), Board::Cell::Color::BLACK))
      m_game.performMove(*move, Board::Cell::Color::BLACK);
    else
      m_game.moveExhausted(Board::Cell::Color::BLACK);
  }

  void MatchAgentPlayer::render()
  {
    m_renderer.draw(m_game.board());
    m_window.display();
  }
}
