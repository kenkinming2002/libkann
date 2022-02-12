#include "MinimaxAgent.hpp"

#include <limits>
#include <assert.h>

#include <iostream>

static std::pair<double, std::optional<Board::Move>> minimax(Board& board, Board::Cell::Color color, size_t depth)
{
  auto moves = board.enumerateMove(color);

  double bestScore = -std::numeric_limits<double>::infinity();
  std::optional<Board::Move> bestMove;

  size_t i = 0;
  for(const auto& move : moves)
  {
    std::cout << "Depth:" << depth << ", i:" << i++ << "/" << moves.size() << '\n';

    const auto cell = board.performMove(move);

    if(cell.type == Board::Cell::Type::GENERAL)
    {
      // We capture the enemy general, and thus we win
      bestScore = std::numeric_limits<double>::infinity();
      bestMove = move;
    }
    else
    {
      double score;
      if(depth == 0)
      {
        const auto [scoreRed, scoreBlack] = board.estimateScore();
        score = color == Board::Cell::Color::RED
          ? scoreRed - scoreBlack
          : scoreBlack - scoreRed;
      }
      else
      {
        const auto opponentColor = color == Board::Cell::Color::RED
          ? Board::Cell::Color::BLACK
          : Board::Cell::Color::RED;
        const auto [opponentScore, _] = minimax(board, opponentColor, depth - 1);
        score = -opponentScore;
      }

      if(bestScore < score)
      {
        bestScore = score;
        bestMove = move;
      }
    }

    board.undoMove(move, cell);
  }

  return {bestScore, *bestMove};
}

Board::Move MinimaxAgent::selectMove(const Board& board, Board::Cell::Color color)
{
  Board copy = board;

  auto [bestScore, bestMove] = minimax(copy, color, 3);
  assert(bestMove);
  return *bestMove;
}
