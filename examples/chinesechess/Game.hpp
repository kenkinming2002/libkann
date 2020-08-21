#pragma once

#include "Board.hpp"

#include <vector>

class Game
{
public:
  void performMove(Board::Move move, Board::Cell::Color color);
  bool undoMove();

public:
  /*
   * Declare that the specific color is unable to make any move, and set the
   * winning color correspondingly.
   */
  void moveExhausted(Board::Cell::Color color);
  bool ended() const { return m_winningColor != Board::Cell::Color::NONE; }
  Board::Cell::Color winningColor() const { return m_winningColor; }

public:
  bool draw() const;

public:
  const auto& board() { return m_board; }

private:
  Board m_board;

private:
  Board::Cell::Color m_winningColor = Board::Cell::Color::NONE;
  std::vector<std::pair<Board::Move, Board::Cell>> m_history;
};
