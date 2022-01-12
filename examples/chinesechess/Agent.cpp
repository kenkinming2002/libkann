#include "Agent.hpp"

#include <cereal/details/helpers.hpp>
#include <cereal/archives/binary.hpp>

#include <stdexcept>
#include <fstream>
#include <filesystem>

Agent::Agent(std::shared_ptr<kann::Model> model) : m_model(std::move(model)) {}

void Agent::loadFromFile(std::filesystem::path filePath)
{
  std::ifstream inputFile;
  inputFile.open(filePath.c_str(), std::ifstream::binary);
  if(!inputFile)
    throw std::runtime_error(std::string("Failed to open file ") + filePath.c_str());

  cereal::BinaryInputArchive inputArchive(inputFile);
  inputArchive(*this);

  m_filePath = std::filesystem::is_symlink(filePath) ? std::filesystem::read_symlink(filePath) : filePath;
}

void Agent::saveToFile(std::filesystem::path filePath) const
{
  if(!m_filePath.empty())
  {
    std::filesystem::create_symlink(m_filePath, filePath);
    return;
  }

  std::ofstream outputFile;
  outputFile.open(filePath.c_str(), std::ofstream::binary);
  if(!outputFile)
    throw std::runtime_error(std::string("Failed to open file ") + filePath.c_str());

  cereal::BinaryOutputArchive outputArchive(outputFile);
  outputArchive(*this);

  m_filePath = std::move(filePath);
}

std::optional<Board::Move> Agent::selectMove(Board board, Board::Cell::Color color)
{
  auto possibleMoves = board.enumerateMove(color);
  if(possibleMoves.empty())
    return std::nullopt;

  auto scores = std::vector<double>(possibleMoves.size());

  for(size_t i=0; i<possibleMoves.size(); ++i)
  {
    const auto& move = possibleMoves[i];

    auto cell = board.performMove(move);
    scores[i] = this->evaluateBoard(board, color);
    board.undoMove(move, cell);
  }

  auto index = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
  return possibleMoves[index];
}

double Agent::evaluateBoard(const Board& board, Board::Cell::Color color)
{
  // TODO: optimize this tight loop
  Eigen::VectorXd input(INPUT_LAYER_SIZE);
  for(uint8_t y=0; y<Board::HEIGHT; ++y)
    for(uint8_t x=0; x<Board::WIDTH; ++x)
    {
      auto position = Board::Position{x, y};
      auto cell = board.cell(position);
      auto index = (y * Board::WIDTH + x) * 2;
      input(index)   = static_cast<double>(cell.type);
      input(index+1) = cell.color == color ? 1.0 : cell.color == Board::Cell::Color::NONE ? 0.0 : -1.0;
    }

  m_model->input(std::move(input));
  m_output = m_model->feedForward();
  return m_output(0);
}

void Agent::learnFrom(const Board& board, Board::Cell::Color color, bool good)
{
  auto otherColor = color == Board::Cell::Color::RED ? Board::Cell::Color::BLACK : Board::Cell::Color::RED;
  {
    this->evaluateBoard(board, color);

    Eigen::VectorXd expectedOutput(1);
    expectedOutput << (good ? 1.0 : 0.0);
    m_model->outputGradient((m_output-expectedOutput)*2.0);
    m_model->backPropagate();
  }
  {
    this->evaluateBoard(board.flipped(), otherColor);

    Eigen::VectorXd expectedOutput(1);
    expectedOutput << (good ? 1.0 : 0.0);
    m_model->outputGradient((m_output-expectedOutput)*2.0);
    m_model->backPropagate();
  }
}

void Agent::learnFrom(Game& game, double learningRate)
{
  if(game.winningColor() == Board::Cell::Color::NONE)
    return;

  auto winningColor = game.winningColor();
  auto losingColor   = winningColor == Board::Cell::Color::RED ? Board::Cell::Color::BLACK : Board::Cell::Color::RED;
  for(;;)
  {
    learnFrom(game.board(), winningColor, true);
    if(!game.undoMove())
      break;

    learnFrom(game.board(), losingColor, false);
    if(!game.undoMove())
      break;
  }
  m_model->train(learningRate);
}

Agent Agent::cross(const Agent& lhs, const Agent& rhs, std::default_random_engine& engine, double mutationRate)
{
  return Agent(kann::Model::cross(*lhs.m_model, *rhs.m_model, engine, mutationRate));
}

