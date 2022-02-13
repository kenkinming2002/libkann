#include "MinimaxAgent.hpp"

#include <limits>
#include <assert.h>

#include <iostream>

template<typename Callback>
auto scopeGuard(Callback callback)
{
  struct ScopeGuard
  {
    constexpr ScopeGuard(Callback callback) : callback(callback) {}
    constexpr ~ScopeGuard() { callback(); }

    constexpr ScopeGuard(const ScopeGuard&) = delete;
    constexpr ScopeGuard(ScopeGuard&&) = delete;

    constexpr ScopeGuard& operator=(const ScopeGuard&) = delete;
    constexpr ScopeGuard& operator=(ScopeGuard&&) = delete;

    Callback callback;
  };

  return ScopeGuard(callback);
}

static Board::Cell::Color opponentColor(Board::Cell::Color color)
{
  return color == Board::Cell::Color::RED ? Board::Cell::Color::BLACK : Board::Cell::Color::RED;
}

static std::pair<double, std::optional<Board::Move>> maximize(Board& board, Board::Cell::Color color, double maximum, size_t depth)
{
  static constexpr double WIN_SCORE = 1e10;

  if(depth == 0)
  {
    const auto [scoreRed, scoreBlack] = board.estimateScore();
    const double score = color == Board::Cell::Color::RED
      ? scoreRed - scoreBlack
      : scoreBlack - scoreRed;

    return {score, std::nullopt};
  }

  const auto moves = board.enumerateMove(color);
  if(moves.empty())
    return {-WIN_SCORE, std::nullopt}; // We ran out of move

  // Recursive
  double bestScore = -std::numeric_limits<double>::infinity();
  std::optional<Board::Move> bestMove;
  for(const auto& move : moves)
  {
    const auto cell = board.performMove(move);
    const auto guard = scopeGuard([&](){ board.undoMove(move, cell); });

    if(cell.type == Board::Cell::Type::GENERAL)
    {
      bestScore = WIN_SCORE;
      bestMove = move;
    }
    else
    {
      const auto opponentColor = ::opponentColor(color);
      const auto [opponentScore, _] = maximize(board, opponentColor, -bestScore, depth - 1);
      const auto score = -opponentScore;
      if(bestScore < score)
      {
        bestScore = score;
        bestMove = move;
      }
    }

    // Alpha-beta pruning
    if(bestScore >= maximum)
      break;
  }

  return {bestScore, bestMove};
}

Board::Move MinimaxAgent::selectMove(const Board& board, Board::Cell::Color color)
{
  Board copy = board;
  auto [bestScore, bestMove] = maximize(copy, color, std::numeric_limits<double>::infinity(), 5);
  assert(bestMove);
  return *bestMove;
}
