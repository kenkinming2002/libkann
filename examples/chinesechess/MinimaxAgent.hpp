#pragma once

#include "Agent.hpp"

class MinimaxAgent : public Agent
{
public:
  Board::Move selectMove(const Board& board, Board::Cell::Color color) override;
};
