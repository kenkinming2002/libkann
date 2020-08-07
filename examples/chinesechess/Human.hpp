#pragma once

#include "Board.hpp"
#include "Renderer.hpp"

#include <optional>

class Human
{
public:
  bool handleInput(const Renderer& renderer, sf::Event event);
  std::optional<Board::Move> pollMove(const Board& board, Board::Cell::Color color);

public:
  std::optional<Board::Position> m_src;
  std::optional<Board::Move> m_move;
};
