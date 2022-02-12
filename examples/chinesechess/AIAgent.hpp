#pragma once

#include "NewAgent.hpp"

#include <libkann/Model.hpp>

#include <cereal/types/memory.hpp>

class AIAgent : public NewAgent
{
public:
  static AIAgent make();
  static AIAgent cross(const AIAgent& lhs, const AIAgent& rhs, std::default_random_engine& engine, double mutationRate);

public:
  AIAgent() = default;
  AIAgent(std::shared_ptr<kann::Model> model);

public:
  Board::Move selectMove(const Board& board, Board::Cell::Color color) override;

public:
  template<typename Archive>
  void serialize(Archive& archive)
  {
    archive(m_model);
  }

private:
  std::shared_ptr<kann::Model> m_model;
};
