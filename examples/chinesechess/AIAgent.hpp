#pragma once

#include "NewAgent.hpp"

#include <libkann/Model.hpp>

class AIAgent : public NewAgent
{
public:
  static AIAgent make();
  AIAgent(std::shared_ptr<kann::Model> model);

public:
  Board::Move selectMove(const Board& board, Board::Cell::Color color) override;

private:
  std::shared_ptr<kann::Model> m_model;
};
