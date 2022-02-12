#include "Game.hpp"

#include <tuple>

GameResult game(Agent& agent1, Agent& agent2)
{
  Board board;

  struct UndoInfo
  {
    Board::Move move;
    Board::Cell cell;
  };
  std::vector<UndoInfo> undoInfos;

  std::optional<Board::Cell::Color> winner;

  static constexpr size_t MAX_MOVE = 50;
  for(size_t i=0; i<MAX_MOVE; ++i)
  {
    auto f = [&](Agent& agent, Board::Cell::Color color)
    {
      if(board.enumerateMove(color).empty())
      {
        winner = color == Board::Cell::Color::RED
          ? Board::Cell::Color::BLACK
          : Board::Cell::Color::RED;

        return true;
      }

      auto move = agent.selectMove(board, color);
      auto cell = board.performMove(move);
      undoInfos.push_back(UndoInfo{
        .move = move,
        .cell = cell
      });

      if(cell.type == Board::Cell::Type::GENERAL)
      {
        winner = color;
        return true;
      }

      return false;
    };

    if(f(agent1, Board::Cell::Color::RED)) break;
    if(f(agent2, Board::Cell::Color::BLACK)) break;
  }

  double score1, score2;
  if(winner)
  {
    if(*winner == Board::Cell::Color::RED)
    {
      score1 = 100.0;
      score2 = 0.0;
    }
    else
    {
      score1 = 0.0;
      score2 = 1.0;
    }
  }
  else
    std::tie(score1, score2) = board.estimateScore();

  std::vector<Board::Move> moves;
  moves.reserve(undoInfos.size());
  std::transform(undoInfos.begin(), undoInfos.end(), std::back_inserter(moves), [](const UndoInfo& info){
    return info.move;
  });

  return GameResult{
    .moves = std::move(moves),
    .winner = winner,
    .score1 = score1,
    .score2 = score2
  };
}
