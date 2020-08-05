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

    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 0, static_cast<double>(move.src.x) / Board::WIDTH);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 1, static_cast<double>(move.src.y) / Board::HEIGHT);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 2, static_cast<double>(move.dst.x) / Board::WIDTH);
    m_neuralNetwork.input(Board::WIDTH * Board::HEIGHT * 2 + 3, static_cast<double>(move.dst.y) / Board::HEIGHT);

    m_neuralNetwork.feedForward();

    scores[i] = m_neuralNetwork.output(0);
  }

  auto index = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
  auto move = possibleMoves[index];

  return board.performMove(move);
}


void Agent::match(Agent& lhs, Agent& rhs, size_t turnLimit)
{
  static constexpr double WINNING_SCORE = 100.0;
  static constexpr double LOSING_SCORE  = -100.0;
  {
    auto winningColor = Board::Cell::Color::NONE;
    Board board;
    for(size_t i=0; i<turnLimit; ++i)
    {
      if(auto state = lhs.performMove(board, Board::Cell::Color::RED); state != Board::State::UNKNOWN)
      {
        if(state == Board::State::WON)
          winningColor = Board::Cell::Color::RED;
        else
          winningColor = Board::Cell::Color::BLACK;

        break;
      }
      if(auto state = rhs.performMove(board, Board::Cell::Color::BLACK); state != Board::State::UNKNOWN)
      {
        if(state == Board::State::WON)
          winningColor = Board::Cell::Color::BLACK;
        else
          winningColor = Board::Cell::Color::RED;

        break;
      }
    }

    switch(winningColor)
    {
    case Board::Cell::Color::RED:
      lhs.m_score += WINNING_SCORE;
      rhs.m_score += LOSING_SCORE;
      break;
    case Board::Cell::Color::BLACK:
      lhs.m_score += LOSING_SCORE;
      rhs.m_score += WINNING_SCORE;
      break;
    case Board::Cell::Color::NONE:
      auto [redScore, blackScore] = board.estimateScore();
      lhs.m_score += redScore;
      rhs.m_score += blackScore;
      break;
    }
  }

  {
    auto winningColor = Board::Cell::Color::NONE;
    Board board;
    for(size_t i=0; i<turnLimit; ++i)
    {
      if(auto state = rhs.performMove(board, Board::Cell::Color::RED); state != Board::State::UNKNOWN)
      {
        if(state == Board::State::WON)
          winningColor = Board::Cell::Color::RED;
        else
          winningColor = Board::Cell::Color::BLACK;

        break;
      }
      if(auto state = lhs.performMove(board, Board::Cell::Color::BLACK); state != Board::State::UNKNOWN)
      {
        if(state == Board::State::WON)
          winningColor = Board::Cell::Color::BLACK;
        else
          winningColor = Board::Cell::Color::RED;

        break;
      }
    }

    switch(winningColor)
    {
    case Board::Cell::Color::RED:
      rhs.m_score += WINNING_SCORE;
      lhs.m_score += LOSING_SCORE;
      break;
    case Board::Cell::Color::BLACK:
      rhs.m_score += LOSING_SCORE;
      lhs.m_score += WINNING_SCORE;
      break;
    case Board::Cell::Color::NONE:
      auto [redScore, blackScore] = board.estimateScore();
      rhs.m_score += redScore;
      lhs.m_score += blackScore;
      break;
    }
  }
}

template<typename PRNG>
Agent Agent::cross(const Agent& lhs, const Agent& rhs, PRNG& prng, double mutationRate)
{
  return Agent(NeuralNetwork::cross(lhs.m_neuralNetwork, rhs.m_neuralNetwork, prng, mutationRate));
}

template Agent Agent::cross(const Agent& lhs, const Agent& rhs, std::mt19937& prng, double mutationRate);
