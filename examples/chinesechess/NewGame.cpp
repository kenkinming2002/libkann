#include "NewGame.hpp"

void NewGame::run(NewAgent& agent1, NewAgent& agent2)
{
  for(;;)
  {
    auto f = [this](NewAgent& agent, Board::Cell::Color color)
    {
      auto move = agent.selectMove(m_board, color);
      auto cell = m_board.performMove(move);
      m_undoInfos.push(UndoInfo{
        .move = move,
        .cell = cell
      });

      if(cell.type == Board::Cell::Type::GENERAL)
      {
        m_winner = color;
        return true;
      }

      return false;
    };

    if(f(agent1, Board::Cell::Color::RED)) break;
    if(f(agent2, Board::Cell::Color::BLACK)) break;
  }
}
