#pragma once

#include "Board.hpp"
#include "Renderer.hpp"

#include <optional>

class Human
{
public:
  bool handleInput(const Renderer& renderer, const Board& board, const Board::Cell::Color color, sf::Event event);
  std::optional<Board::Move> pollMove(const Board& board, Board::Cell::Color color);

public:
  std::optional<Board::Position> m_src, m_dst;
};
