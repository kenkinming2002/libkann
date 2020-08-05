#include "Agent.hpp"

Agent::Agent(NeuralNetwork neuralNetwork) : m_neuralNetwork(std::move(neuralNetwork)) {}

Board::Cell::Color Agent::performMove(Board& board, Board::Cell::Color color)
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
    return color == Board::Cell::Color::RED ? Board::Cell::Color::BLACK : Board::Cell::Color::RED;

  auto scores = std::vector<double>(possibleMoves.size());

  for(size_t i=0; i<possibleMoves.size(); ++i)
  {
    const auto& move = possibleMoves[i];

    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 0, static_cast<double>(move.src.x) / Board::WIDTH);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 1, static_cast<double>(move.src.y) / Board::HEIGHT);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 2, static_cast<double>(move.dst.x) / Board::WIDTH);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 3, static_cast<double>(move.dst.y) / Board::HEIGHT);

    m_neuralNetwork.feedForward();

    scores[i] = m_neuralNetwork.output(0);
  }

  auto index = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
  auto move = possibleMoves[index];

  return board.performMove(move, color);
}

Agent::MatchSingleResult Agent::matchSingle(Agent& lhs, Agent& rhs, size_t turnLimit)
{
  static constexpr double WINNING_SCORE = 100.0;
  static constexpr double LOSING_SCORE  = -100.0;

  Board board;
  auto winningColor = [&]()
  {
    for(size_t i=0; i<turnLimit; ++i)
    {
      if(auto color = lhs.performMove(board, Board::Cell::Color::RED);
          color != Board::Cell::Color::NONE)
        return color;

      if(auto color = rhs.performMove(board, Board::Cell::Color::BLACK);
          color != Board::Cell::Color::NONE)
        return color;
    }
    return Board::Cell::Color::NONE;
  }();

  switch(winningColor)
  {
  case Board::Cell::Color::RED:
    lhs.m_score += WINNING_SCORE;
    rhs.m_score += LOSING_SCORE;
    return MatchSingleResult{AgentID::_1, board};
  case Board::Cell::Color::BLACK:
    lhs.m_score += LOSING_SCORE;
    rhs.m_score += WINNING_SCORE;
    return MatchSingleResult{AgentID::_2, board};
  case Board::Cell::Color::NONE:
  {
    auto [redScore, blackScore] = board.estimateScore();
    lhs.m_score += redScore;
    rhs.m_score += blackScore;
    auto winningAgent = [&](){
      if(lhs.score() > rhs.score())
        return AgentID::_1;
      if(lhs.score() < rhs.score())
        return AgentID::_2;
      return AgentID::NONE;
    }();
    return MatchSingleResult{winningAgent, board};
  }
  default:
    throw std::runtime_error("I don't know WTH happened");
  }
}

Agent::MatchResult Agent::match(Agent& lhs, Agent& rhs, size_t turnLimit)
{
  auto result1 = matchSingle(lhs, rhs, turnLimit);
  auto result2 = matchSingle(rhs, lhs, turnLimit);

  auto winningAgent = [&]() {
    if(lhs.score() > rhs.score())
      return AgentID::_1;
    if(lhs.score() < rhs.score())
      return AgentID::_2;
    return AgentID::NONE;
  }();

  return MatchResult{winningAgent, result1.board, result2.board};
}

template<typename PRNG>
Agent Agent::cross(const Agent& lhs, const Agent& rhs, PRNG& prng, double mutationRate)
{
  return Agent(NeuralNetwork::cross(lhs.m_neuralNetwork, rhs.m_neuralNetwork, prng, mutationRate));
}

template Agent Agent::cross(const Agent& lhs, const Agent& rhs, std::mt19937& prng, double mutationRate);
