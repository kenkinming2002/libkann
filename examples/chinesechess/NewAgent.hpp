#pragma once

#include "Board.hpp"

class NewAgent
{
public:
  virtual ~NewAgent() = default;

public:
  virtual Board::Move selectMove(const Board& board, Board::Cell::Color color) = 0;

public:
  template<typename Archive>
  void serialize(Archive& archive) {}
};
