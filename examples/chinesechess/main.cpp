#include "Board.hpp"
#include "Agent.hpp"

#include <iostream>
#include <random>
#include <chrono>

template<typename PRNG>
Agent makeAgent(PRNG& prng)
{
  return Agent(NeuralNetwork({Agent::INPUT_LAYER_SIZE, 10, 10, Agent::OUTPUT_LAYER_SIZE}, prng, activation_function::sigmoid));
}

int main()
{
  std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());;

  Board board;
  auto agent1 = makeAgent(engine), agent2 = makeAgent(engine);

  for(size_t i=0; i<1000; ++i)
  {
    if(agent1.performMove(board, Board::Cell::Color::RED) != Board::State::UNKNOWN)
      break;
    if(agent2.performMove(board, Board::Cell::Color::BLACK) != Board::State::UNKNOWN)
      break;
    std::cout << "\r" << i++ << std::flush;
  }
  std::cout << "\r" << board;
}
