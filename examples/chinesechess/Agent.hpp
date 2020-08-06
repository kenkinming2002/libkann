#pragma ocne

#include "Board.hpp"

#include <libkann/NeuralNetwork.hpp>


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
  enum class MoveResult { WON, LOST, NONE };
  Board::Cell::Color performMove(Board& board, Board::Cell::Color color);

public:
  enum class AgentID { NONE, _1, _2};
  struct MatchSingleResult
  {
    AgentID winningAgent;
    Board board;
  };
  static MatchSingleResult matchSingle(Agent& lhs, Agent& rhs, size_t turnLimit = 100000);

  struct MatchResult
  {
    AgentID winningAgent;
    Board board1;
    Board board2;
  };
  static MatchResult match(Agent& lhs, Agent& rhs, size_t turnLimit = 100000);

  template<typename PRNG>
  static Agent cross(const Agent& lhs, const Agent& rhs, PRNG& prng, double mutationRate);

public:
  auto clearScore() { m_score = 0.0; }
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

