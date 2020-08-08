#include "Human.hpp"

#include <SFML/Window/Event.hpp>
#include <optional>

bool Human::handleInput(const Renderer& renderer, const Board& board, const Board::Cell::Color color, sf::Event event)
{
  if(event.type == sf::Event::MouseButtonPressed)
  {
    if(auto boardCoords = renderer.screenToBoardCoords(sf::Vector2i(event.mouseButton.x, event.mouseButton.y)))
    {
      if(board.cell(*boardCoords).color != color)
        m_dst = boardCoords;
      else
        m_src = boardCoords;
    }

    return true;
  }
  else
    return false;
}


std::optional<Board::Move> Human::pollMove(const Board& board, Board::Cell::Color color)
{
  if(m_src && m_dst)
  {
    auto move = Board::Move{*m_src, *m_dst};
    if(board.validateMove(move, color))
      return move;
    else
      m_dst = std::nullopt;
  }
  return std::nullopt;
}

