#pragma once

#include "Board.hpp"

#include <stack>

class Game
{
public:
  void performMove(Board::Move move, Board::Cell::Color color);
  void undoMove();

public:
  /*
   * Declare that the specififc color is unable to make any move, and set the
   * winning color correspondingly.
   */
  void moveExhausted(Board::Cell::Color color);
  bool ended() const { return m_winningColor != Board::Cell::Color::NONE; }

public:
  const auto& board() { return m_board; }

private:
  Board m_board;

private:
  Board::Cell::Color m_winningColor = Board::Cell::Color::NONE;
  std::stack<std::pair<Board::Move, Board::Cell>> m_history;
};
