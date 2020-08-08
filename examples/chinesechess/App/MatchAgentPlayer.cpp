#include "MatchAgentPlayer.hpp"

#include <SFML/Window/Event.hpp>

static constexpr unsigned WINDOW_WIDTH = 1000;
static constexpr unsigned WINDOW_HEIGHT = 1000;

namespace App
{
  MatchAgentPlayer::MatchAgentPlayer(const char* agentFilePath) 
    : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), ""),
      m_renderer(m_window),
      m_agent(Agent::loadFromFile(agentFilePath)) {}

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
      if(m_human.handleInput(m_renderer, m_board, Board::Cell::Color::RED, event))
        continue;

      if(event.type == sf::Event::Closed)
        m_window.close();
    }
  }

  void MatchAgentPlayer::update()
  {
    if(m_matchEnded)
      return;

    {
      auto move = m_human.pollMove(m_board, Board::Cell::Color::RED);
      if(!move)
        return;

      if(auto color = m_board.performMove(*move, Board::Cell::Color::RED); color != Board::Cell::Color::NONE)
      {
        m_matchEnded = true;
        return;
      }
    }

    {
      auto move = m_agent.selectMove(m_board, Board::Cell::Color::BLACK);
      if(!move)
      {
        m_matchEnded = true;
        return;
      }

      if(auto color = m_board.performMove(*move, Board::Cell::Color::BLACK); color != Board::Cell::Color::NONE)
      {
        m_matchEnded = true;
        return;
      }
    }
  }

  void MatchAgentPlayer::render()
  {
    m_renderer.draw(m_board);
    m_window.display();
  }
}
