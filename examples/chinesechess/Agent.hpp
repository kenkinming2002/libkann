#pragma once

#include "Board.hpp"

#include <libkann/NeuralNetwork.hpp>
#include <optional>

class Agent
{
public:
  static constexpr size_t INPUT_LAYER_SIZE = Board::WIDTH * Board::HEIGHT * 2 + 4; /* board states and the move*/
  static constexpr size_t OUTPUT_LAYER_SIZE = 1; /* score of the move */

public:
  Agent() = default;
  Agent(NeuralNetwork neuralNetwork);

public:
  auto& neuralNetwork() { return m_neuralNetwork; }
  const auto& neuralNetwork() const { return m_neuralNetwork; }

public:
  std::optional<Board::Move> selectMove(const Board& board, Board::Cell::Color color);

public:
  template<typename PRNG>
  static Agent cross(const Agent& lhs, const Agent& rhs, PRNG& prng, double mutationRate);

public:
  void addScore(double value) { m_score += value; }
  auto clearScore() { m_score = 0.0; }

public:
  auto score() const { return m_score; }

public:
  template<typename Archive>
  void serialize(Archive& archive)
  {
    archive(m_neuralNetwork);
  }

private:
  NeuralNetwork m_neuralNetwork;
  double m_score = 0.0;
};

