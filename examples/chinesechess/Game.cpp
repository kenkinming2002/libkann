#include "Game.hpp"

void Game::performMove(Board::Move move)
{
  auto cell = m_board.performMove(move);
  m_history.emplace(move, cell);
  if(cell.type == Board::Cell::Type::GENERAL)
    m_ended = true;
}

void Game::undoMove()
{
  if(!m_history.empty())
  {
    auto [move, cell] = m_history.top();
    m_board.undoMove(move, cell);
    m_history.pop();
    if(cell.type == Board::Cell::Type::GENERAL)
      m_ended = false;
  }
}
