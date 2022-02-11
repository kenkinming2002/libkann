#pragma once

#include "NewAgent.hpp"
#include "Renderer.hpp"

#include <libkann/Model.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>

class HumanAgent : public NewAgent
{
public:
  HumanAgent();

public:
  Board::Move selectMove(const Board& board, Board::Cell::Color color) override;

private:
  void loop(std::stop_token token);

private:
  sf::RenderWindow m_window;
  Renderer m_renderer;

  std::mutex m_mutexBoard;
  Board m_board;

  std::mutex m_mutexMove;
  std::condition_variable m_cvMove;
  std::optional<Board::Move> m_move;

  std::jthread m_thread;
};
