#include "Game.hpp"

#include <cassert>
#include <algorithm>
#include <iostream>

void Game::performMove(Board::Move move, Board::Cell::Color color)
{
  assert(!this->ended());
  assert(color != Board::Cell::Color::NONE);

  auto cell = m_board.performMove(move);
  m_history.emplace_back(move, cell);
  if(cell.type == Board::Cell::Type::GENERAL)
    m_winningColor = color;
}

bool Game::undoMove()
{
  if(!m_history.empty())
  {
    auto [move, cell] = m_history.back();
    m_board.undoMove(move, cell);
    m_history.pop_back();
    if(cell.type == Board::Cell::Type::GENERAL)
      m_winningColor = Board::Cell::Color::NONE;

    return true;
  }
  else
    return false;
}

void Game::moveExhausted(Board::Cell::Color color)
{
  assert(!this->ended());
  assert(color != Board::Cell::Color::NONE);

  m_winningColor = color == Board::Cell::Color::RED ? Board::Cell::Color::BLACK : Board::Cell::Color::RED;
}

bool Game::draw() const
{
  if(m_history.size() < 4)
    return false;

  auto complementary = [](auto val1, auto val2){
    auto [move1, cell1] = val1;
    auto [move2, cell2] = val2;
    return move1.src == move2.dst && move1.dst == move2.src;
  };

  auto benign = [](auto val){
    auto [move, cell] = val;
    return cell.empty();
  };
  
  return std::all_of(std::prev(m_history.end(), 4), m_history.end(), benign) &&
    complementary(*std::prev(m_history.end(), 1), *std::prev(m_history.end(), 3)) &&
    complementary(*std::prev(m_history.end(), 2), *std::prev(m_history.end(), 4));
}
