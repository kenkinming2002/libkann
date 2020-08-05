#include "Board.hpp"
#include "Agent.hpp"

#include <iostream>
#include <random>
#include <chrono>

template<typename PRNG>
Agent makeAgent(PRNG& prng)
{
  auto neuralNetwork = NeuralNetwork({Agent::INPUT_LAYER_SIZE, 31, 31, 37, Agent::OUTPUT_LAYER_SIZE}, prng, activation_function::sigmoid);
  return Agent(std::move(neuralNetwork));
}

int main()
{
  std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  Board board;
  auto agent1 = makeAgent(engine);
  auto agent2 = makeAgent(engine);
  Agent::match(agent1, agent2, 100);
  std::cout << agent1.score() << "," << agent2.score() << '\n';
}
