#include "MatchAgents.hpp"

#include <SFML/Window/Event.hpp>
#include <SFML/Window/Keyboard.hpp>

static constexpr unsigned WINDOW_WIDTH = 1000;
static constexpr unsigned WINDOW_HEIGHT = 1000;

namespace App
{
  MatchAgents::MatchAgents(const char* agent1FileName, const char* agent2FileName) 
    : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), ""), m_renderer(m_window)
  {
    m_agent1.loadFromFile(agent1FileName);
    m_agent2.loadFromFile(agent2FileName);
  }

  void MatchAgents::run()
  {
    while(this->loop());
  }

  bool MatchAgents::loop()
  {
    this->handleInput();
    this->update();
    this->render();
    return m_window.isOpen();
  }

  void MatchAgents::handleInput()
  {
    sf::Event event;
    while (m_window.pollEvent(event))
    {
      if(event.type == sf::Event::KeyPressed)
      {
        switch(event.key.code)
        {
        case sf::Keyboard::Left:
          m_game.undoMove();
          m_currentAgent = m_currentAgent == AgentID::_1 ? AgentID::_2 : AgentID::_1;
          break;
        case sf::Keyboard::Right:
          switch(m_currentAgent)
          {
            case AgentID::_1:
              if(auto move = m_agent1.selectMove(m_game.board(), Board::Cell::Color::RED))
                m_game.performMove(*move);
              else
                m_game.end();
              break;
            case AgentID::_2:
              if(auto move = m_agent2.selectMove(m_game.board(), Board::Cell::Color::BLACK))
                m_game.performMove(*move);
              else
                m_game.end();
              break;
          }
          m_currentAgent = m_currentAgent == AgentID::_1 ? AgentID::_2 : AgentID::_1;
        default:
          break;
        }
      }
      
      if(event.type == sf::Event::Closed)
        m_window.close();
    }
  }

  void MatchAgents::update()
  {
    // we don't do any update here
  }

  void MatchAgents::render()
  {
    m_renderer.draw(m_game.board());
    m_window.display();
  }
}
