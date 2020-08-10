#include "Game.hpp"

#include <cassert>

void Game::performMove(Board::Move move, Board::Cell::Color color)
{
  assert(!this->ended());
  assert(color != Board::Cell::Color::NONE);

  auto cell = m_board.performMove(move);
  m_history.emplace(move, cell);
  if(cell.type == Board::Cell::Type::GENERAL)
    m_winningColor = color;
}

void Game::undoMove()
{
  if(!m_history.empty())
  {
    auto [move, cell] = m_history.top();
    m_board.undoMove(move, cell);
    m_history.pop();
    if(cell.type == Board::Cell::Type::GENERAL)
      m_winningColor = Board::Cell::Color::NONE;
  }
}

void Game::moveExhausted(Board::Cell::Color color)
{
  assert(!this->ended());
  assert(color != Board::Cell::Color::NONE);

  m_winningColor = color == Board::Cell::Color::RED ? Board::Cell::Color::BLACK : Board::Cell::Color::RED;
}
