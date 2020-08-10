#include "Agent.hpp"

#include <cereal/details/helpers.hpp>
#include <stdexcept>
#include <fstream>
#include <cereal/archives/binary.hpp>

Agent::Agent(NeuralNetwork neuralNetwork) : m_neuralNetwork(std::move(neuralNetwork)) {}

void Agent::loadFromFile(const char* fileName)
{
  std::ifstream inputFile;
  inputFile.open(fileName, std::ifstream::binary);
  if(!inputFile)
    throw std::runtime_error(std::string("Failed to open file ") + fileName);

  cereal::BinaryInputArchive inputArchive(inputFile);
  inputArchive(*this);
}

void Agent::saveToFile(const char* fileName) const
{
  std::ofstream outputFile;
  outputFile.open(fileName, std::ofstream::binary);
  if(!outputFile)
    throw std::runtime_error(std::string("Failed to open file ") + fileName);

  cereal::BinaryOutputArchive outputArchive(outputFile);
  outputArchive(*this);
}

std::optional<Board::Move> Agent::selectMove(const Board& board, Board::Cell::Color color)
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
    return std::nullopt;

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
  return possibleMoves[index];
}

template<typename PRNG>
Agent Agent::cross(const Agent& lhs, const Agent& rhs, PRNG& prng, double mutationRate)
{
  return Agent(NeuralNetwork::cross(lhs.m_neuralNetwork, rhs.m_neuralNetwork, prng, mutationRate));
}

template Agent Agent::cross(const Agent& lhs, const Agent& rhs, std::mt19937& prng, double mutationRate);
