#include "Human.hpp"

#include <stdexcept>
#include <iostream>

#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Event.hpp>

bool Human::handleInput(const Renderer& renderer, sf::Event event)
{
  if(event.type == sf::Event::MouseButtonPressed)
  {
    if(auto boardCoords = renderer.screenToBoardCoords(sf::Vector2i(event.mouseButton.x, event.mouseButton.y)))
    {
      if(!m_src)
        m_src = boardCoords;
      else
        m_move = Board::Move{*m_src, *boardCoords};
    }
    return true;
  }
  else
    return false;
}


std::optional<Board::Move> Human::pollMove(const Board& board, Board::Cell::Color color)
{
  if(m_move)
  {
    if(board.validateMove(*m_move, color))
    {
      return m_move;
    }
    else
    {
      m_src = std::nullopt;
      m_move = std::nullopt;
    }
  }
  return std::nullopt;
}

