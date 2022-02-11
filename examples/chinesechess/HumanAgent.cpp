#include "HumanAgent.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <functional>

static constexpr unsigned WINDOW_WIDTH = 1000;
static constexpr unsigned WINDOW_HEIGHT = 1000;

HumanAgent::HumanAgent()
  : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Human Agent"),
    m_renderer(m_window),
    m_thread(std::bind(&HumanAgent::loop, this, std::placeholders::_1)) {}

Board::Move HumanAgent::selectMove(const Board& board, Board::Cell::Color color)
{
  // Update board
  {
    std::lock_guard guard(m_mutexBoard);
    m_board = board;
  }

  Board::Move move;
  for(;;)
  {
    std::unique_lock lock(m_mutexMove);
    m_cvMove.wait(lock, [this](){
      return m_move.has_value();
    });

    move = *m_move;
    lock.unlock();

    if(board.validateMove(move, color))
      break;
  }

  // Update board
  {
    std::lock_guard guard(m_mutexBoard);
    m_board.performMove(move);
  }

  return move;
}

void HumanAgent::loop(std::stop_token token)
{
  std::optional<Board::Position> src, dst;
  for(;;)
  {
    if(token.stop_requested())
      goto finish;

    sf::Event evnt;
    while(m_window.pollEvent(evnt))
    {
      switch(evnt.type)
      {
      case sf::Event::Closed:
        goto finish;
      case sf::Event::MouseButtonPressed:
        if(auto coords = m_renderer.screenToBoardCoords({evnt.mouseButton.x, evnt.mouseButton.y}))
        {
          if(!src)
            src = coords;

          if(!dst)
            dst = coords;

          if(src && dst)
          {
            {
              std::lock_guard guard(m_mutexMove);
              m_move = Board::Move{
                .src = *src,
                .dst = *dst
              };
            }
            m_cvMove.notify_one();

            src.reset();
            dst.reset();
          }
        }
        break;
      default:
        break;
      }
    }

    m_renderer.draw(m_board);
    m_window.display();
  }

finish:;
}
