#pragma once

#include "Board.hpp"

#include <stack>

class Game
{
public:
  void performMove(Board::Move move);
  void undoMove();

public:
  const auto& board() { return m_board; }

public:
  void end() { m_ended = true; }
  [[nodiscard]] bool ended() const { return m_ended; }

private:
  Board m_board;

private:
  bool m_ended = false;
  std::stack<std::pair<Board::Move, Board::Cell>> m_history;
};
