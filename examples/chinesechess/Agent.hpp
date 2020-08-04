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

private:
  NeuralNetwork m_neuralNetwork;
};
