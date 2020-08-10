#include "Match.hpp"

#include "Game.hpp"

MatchResult matchSingle(Agent& lhs, Agent& rhs, size_t turnLimit)
{
  static constexpr double WINNING_SCORE = 100.0;
  static constexpr double LOSING_SCORE  = -100.0;

  Game game;
  for(size_t i=0; i<turnLimit; ++i)
  {
    if(auto move = lhs.selectMove(game.board(), Board::Cell::Color::RED))
      game.performMove(*move, Board::Cell::Color::RED);
    else
      game.moveExhausted(Board::Cell::Color::RED);

    if(game.ended() || game.draw())
      break;

    if(auto move = rhs.selectMove(game.board(), Board::Cell::Color::BLACK))
      game.performMove(*move, Board::Cell::Color::BLACK);
    else
      game.moveExhausted(Board::Cell::Color::BLACK);

    if(game.ended() || game.draw())
      break;
  }

  switch(game.winningColor())
  {
  case Board::Cell::Color::RED:
    return MatchResult{WINNING_SCORE, LOSING_SCORE};
  case Board::Cell::Color::BLACK:
    return MatchResult{LOSING_SCORE, WINNING_SCORE};
  case Board::Cell::Color::NONE:
    auto [redScore, blackScore] = game.board().estimateScore();
    return MatchResult{redScore, blackScore};
  }

  assert(false);
  __builtin_unreachable();
}

MatchResult match(Agent& lhs, Agent& rhs, size_t turnLimit)
{
  auto result1 = matchSingle(lhs, rhs, turnLimit);
  auto result2 = matchSingle(rhs, lhs, turnLimit);
  return MatchResult{result1.score1 + result2.score2, result1.score2 + result2.score1};
}
