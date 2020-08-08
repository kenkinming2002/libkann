#include "Match.hpp"

MatchSingleResult matchSingle(Agent& lhs, Agent& rhs, size_t turnLimit)
{
  static constexpr double WINNING_SCORE = 100.0;
  static constexpr double LOSING_SCORE  = -100.0;

  Board board;
  auto winningColor = [&]()
  {
    for(size_t i=0; i<turnLimit; ++i)
    {
      {
        auto move = lhs.selectMove(board, Board::Cell::Color::RED);
        if(!move)
          return Board::Cell::Color::BLACK;
        if(auto color = board.performMove(*move, Board::Cell::Color::RED); color != Board::Cell::Color::NONE)
          return color;
      }
      {
        auto move = rhs.selectMove(board, Board::Cell::Color::BLACK);
        if(!move)
          return Board::Cell::Color::RED;
        if(auto color = board.performMove(*move, Board::Cell::Color::BLACK); color != Board::Cell::Color::NONE)
          return color;
      }
    }
    return Board::Cell::Color::NONE;
  }();

  switch(winningColor)
  {
  case Board::Cell::Color::RED:
    lhs.addScore(WINNING_SCORE);
    rhs.addScore(LOSING_SCORE);
    return MatchSingleResult{PlayerID::AGENT1, board};
  case Board::Cell::Color::BLACK:
    lhs.addScore(LOSING_SCORE);
    rhs.addScore(WINNING_SCORE);
    return MatchSingleResult{PlayerID::AGENT2, board};
  case Board::Cell::Color::NONE:
  {
    auto [redScore, blackScore] = board.estimateScore();
    lhs.addScore(redScore);
    rhs.addScore(blackScore);
    auto winningAgent = [&](){
      if(lhs.score() > rhs.score())
        return PlayerID::AGENT1;
      if(lhs.score() < rhs.score())
        return PlayerID::AGENT2;
      return PlayerID::NONE;
    }();
    return MatchSingleResult{winningAgent, board};
  }
  default:
    throw std::runtime_error("I don't know WTH happened");
  }
}

MatchResult match(Agent& lhs, Agent& rhs, size_t turnLimit)
{
  auto result1 = matchSingle(lhs, rhs, turnLimit);
  auto result2 = matchSingle(rhs, lhs, turnLimit);

  auto winningAgent = [&]() {
    if(lhs.score() > rhs.score())
      return PlayerID::AGENT1;
    if(lhs.score() < rhs.score())
      return PlayerID::AGENT2;
    return PlayerID::NONE;
  }();

  return MatchResult{winningAgent, result1.board, result2.board};
}
