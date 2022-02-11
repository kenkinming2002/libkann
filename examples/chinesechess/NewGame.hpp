#pragma once

#include "Board.hpp"
#include "NewAgent.hpp"

#include <stack>

class NewGame
{
public:
  void run(NewAgent& agent1, NewAgent& agent2);

public:
  Board::Cell::Color winner() const;

private:
  Board m_board;

  struct UndoInfo
  {
    Board::Move move;
    Board::Cell cell;
  };
  std::stack<UndoInfo> m_undoInfos;

  Board::Cell::Color m_winner;
};

