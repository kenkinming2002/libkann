#pragma ocne

#include "Board.hpp"

#include <libkann/NeuralNetwork.hpp>


class Agent
{
public:
  static constexpr size_t INPUT_LAYER_SIZE = Board::WIDTH * Board::HEIGHT * 2 + 4; /* board states and the move*/
  static constexpr size_t OUTPUT_LAYER_SIZE = 1; /* score of the move */

public:
  Agent(NeuralNetwork neuralNetwork);

public:
  Board::State performMove(Board& board, Board::Cell::Color color);

public:
  static void match(Agent& lhs, Agent& rhs, size_t turnLimit = 100000);

public:
  auto score() const { return m_score; }

private:
  NeuralNetwork m_neuralNetwork;
  double m_score = 0.0;
};

