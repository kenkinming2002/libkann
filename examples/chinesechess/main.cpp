#include "Board.hpp"
#include "Agent.hpp"

#include <iostream>
#include <random>
#include <chrono>

Agent makeAgent()
{
  std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count()); // Have to use a separate seed because repetition problem is so serious.
  auto neuralNetwork = NeuralNetwork({Agent::INPUT_LAYER_SIZE, 31, 31, 37, Agent::OUTPUT_LAYER_SIZE}, engine, activation_function::sigmoid);
  return Agent(std::move(neuralNetwork));
}

int main()
{
  Board board;
  auto agent1 = makeAgent();
  auto agent2 = makeAgent();
  Agent::match(agent1, agent2, 100);
  std::cout << agent1.score() << "," << agent2.score() << '\n';
}
