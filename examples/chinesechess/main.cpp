#include "Board.hpp"

#include <iostream>
#include <random>
#include <chrono>

int main()
{
  std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());;
  Board board;

  auto performMove = [&](Board::Cell::Color color) -> Board::State {
    auto possibleMoves = board.enumerateMove(color);
    if(possibleMoves.empty())
      return Board::State::LOST;

    std::uniform_int_distribution<size_t> indexDistribution(0, possibleMoves.size()-1);
    auto move = possibleMoves[indexDistribution(engine)];
    return board.performMove(move);
  };

  for(;;)
  {
    if(performMove(Board::Cell::Color::RED) != Board::State::UNKNOWN)
      break;
    std::cout << board << "=========\n";

    if(performMove(Board::Cell::Color::RED) != Board::State::UNKNOWN)
      break;
    std::cout << board << "=========\n";
  }
  std::cout << board << "=========\n";
}
