#include "AIAgent.hpp"
#include "libkann/Model.hpp"

#include <libkann/Loader.hpp>

#include <limits>

AIAgent AIAgent::make(std::default_random_engine& engine)
{
  static auto layer = kann::loadLayer("examples/chinesechess/agent.yaml");
  auto model = std::make_shared<kann::Model>(layer->clone());
  model->randomize(engine);
  return AIAgent(std::move(model));
}

AIAgent AIAgent::cross(const AIAgent& lhs, const AIAgent& rhs, std::default_random_engine& engine, double mutationRate)
{
  auto model = kann::cross(*lhs.m_model, *rhs.m_model, engine, mutationRate);
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

    auto input = std::make_shared<kann::Tensor>(Board::WIDTH * Board::HEIGHT * 7);
    const Board::Cell::Type types[] = {
      Board::Cell::Type::CHARIOT,
      Board::Cell::Type::KNIGHT,
      Board::Cell::Type::ELEPHANT,
      Board::Cell::Type::ADVISOR,
      Board::Cell::Type::GENERAL,
      Board::Cell::Type::CANNON,
      Board::Cell::Type::SOLDIER
    };
    for(size_t i=0; i<sizeof types / sizeof types[0]; ++i)
      for(uint8_t y=0; y<Board::HEIGHT; ++y)
        for(uint8_t x=0; x<Board::WIDTH; ++x)
        {
          auto position = Board::Position{x, y};
          auto cell = copy.cell(position);
          auto index = i * Board::HEIGHT * Board::WIDTH + y * Board::WIDTH + x;
          input->asArray()(index) =
            cell.type == types[i]                  ? 0.0 :
            cell.color == color                    ? 1.0 :
            cell.color == Board::Cell::Color::NONE ? 0.0 :
            -1.0;
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
