#include "Agent.hpp"

Agent::Agent(NeuralNetwork neuralNetwork) : m_neuralNetwork(std::move(neuralNetwork)) {}

Board::State Agent::performMove(Board& board, Board::Cell::Color color)
{
  for(uint8_t y=0; y<Board::HEIGHT; ++y)
    for(uint8_t x=0; x<Board::WIDTH; ++x)
    {
      auto position = Board::Position{x, y};
      auto cell = board.cell(position);
      auto index = (y * Board::WIDTH + x) * 2;

      m_neuralNetwork.input(index+0, static_cast<double>(cell.type));
      m_neuralNetwork.input(index+1, static_cast<double>(cell.color));
    }

  auto possibleMoves = board.enumerateMove(color);
  if(possibleMoves.empty())
    return Board::State::LOST;

  auto scores = std::vector<double>(possibleMoves.size());

  for(size_t i=0; i<possibleMoves.size(); ++i)
  {
    const auto& move = possibleMoves[i];
    auto& score = scores[i];

    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 0, static_cast<double>(move.src.x) / Board::WIDTH);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 1, static_cast<double>(move.src.y) / Board::HEIGHT);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 2, static_cast<double>(move.dst.x) / Board::WIDTH);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 3, static_cast<double>(move.dst.y) / Board::HEIGHT);

    m_neuralNetwork.feedForward();

    score = m_neuralNetwork.output(0);
  }

  auto index = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
  auto move = possibleMoves[index];

  return board.performMove(move);
}
