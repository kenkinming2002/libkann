#include "AIAgent.hpp"

#include <libkann/Loader.hpp>

#include <limits>

AIAgent AIAgent::make()
{
  static auto layer = kann::loadLayer("examples/chinesechess/agent.yaml");
  auto model = std::make_shared<kann::Model>(layer);
  model->randomize();
  return AIAgent(std::move(model));
}

AIAgent::AIAgent(std::shared_ptr<kann::Model> model)
  : m_model(std::move(model)) {}

Board::Move AIAgent::selectMove(const Board& board, Board::Cell::Color color)
{
  auto moves = board.enumerateMove(color);

  Board::Move bestMove;
  double bestScore = std::numeric_limits<double>::min();
  for(const auto& move : moves)
  {
    auto copy = board;
    copy.performMove(move);

    auto input = std::make_shared<kann::Tensor>(Board::WIDTH * Board::HEIGHT * 2);
    for(uint8_t y=0; y<Board::HEIGHT; ++y)
      for(uint8_t x=0; x<Board::WIDTH; ++x)
      {
        auto position = Board::Position{x, y};
        auto cell = copy.cell(position);
        auto index = (y * Board::WIDTH + x) * 2;
        input->asArray()(index)   = static_cast<double>(cell.type);
        input->asArray()(index+1) = cell.color == color ? 1.0 : cell.color == Board::Cell::Color::NONE ? 0.0 : -1.0;
      }

    auto output = m_model->predict(input);
    double score = output->asScalar();

    if(bestScore < score)
    {
      bestMove = move;
      bestScore = score;
    }
  }

  return bestMove;
}
