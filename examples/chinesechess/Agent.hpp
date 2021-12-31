#pragma once

#include "Board.hpp"
#include "Game.hpp"

#include <libkann/Model.hpp>

#include <optional>
#include <filesystem>

class Agent
{
public:
  static constexpr size_t INPUT_LAYER_SIZE = Board::WIDTH * Board::HEIGHT * 2; /* board states */
  static constexpr size_t OUTPUT_LAYER_SIZE = 1; /* score of the boatd */

public:
  Agent() = default;
  Agent(kann::Model model);

public:
  void loadFromFile(std::filesystem::path filePath);
  void saveToFile(std::filesystem::path filePath) const;

public:
  auto& model() { return m_model; }
  const auto& model() const { return m_model; }

public:
  std::optional<Board::Move> selectMove(Board board, Board::Cell::Color color);
  double evaluateBoard(const Board& board, Board::Cell::Color color);

public:
  void learnFrom(const Board& board, Board::Cell::Color color, bool good);
  void learnFrom(Game& game, double learnigRate);

public:
  static Agent cross(const Agent& lhs, const Agent& rhs, std::default_random_engine& engine, double mutationRate);

public:
  void addScore(double value) { m_score += value; }
  auto clearScore() { m_score = 0.0; }

public:
  auto score() const { return m_score; }

public:
  template<typename Archive>
  void serialize(Archive& archive)
  {
    archive(m_model);
  }

private:
  kann::Model m_model;
  double m_score = 0.0;

private:
  mutable std::filesystem::path m_filePath;
};

