#include "NewGame.hpp"

GameResult game(NewAgent& agent1, NewAgent& agent2)
{
  Board board;

  struct UndoInfo
  {
    Board::Move move;
    Board::Cell cell;
  };
  std::stack<UndoInfo> undoInfos;

  std::optional<Board::Cell::Color> winner;

  for(;;)
  {
    auto f = [&](NewAgent& agent, Board::Cell::Color color)
    {
      auto move = agent.selectMove(board, color);
      auto cell = board.performMove(move);
      undoInfos.push(UndoInfo{
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

  auto [score1, score2] = board.estimateScore();
  return GameResult{
    .winner = winner,
    .score1 = score1,
    .score2 = score2
  };
}
