#pragma once

#include "Board.hpp"
#include "NewAgent.hpp"

#include <optional>
#include <stack>

struct GameResult
{
  std::vector<Board::Move> moves;
  std::optional<Board::Cell::Color> winner;
  double score1, score2;
};
GameResult game(NewAgent& agent1, NewAgent& agent2);
