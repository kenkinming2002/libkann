#include "App.hpp"

#include "Config.hpp"

#include <iostream>
#include <algorithm>

App::App(seed_type seed) : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "App"),
  m_world(seed),
  m_renderer(m_window, sf::View({0.0, 0.0}, {WINDOW_WIDTH, WINDOW_HEIGHT})),
  m_renderTimes{0.0f} {}

void App::run()
{
  while(this->loop());
}

bool App::loop()
{
  handleInput();
  update();
  render();
  return m_window.isOpen();
}

void App::handleInput()
{
  sf::Event event;
  while (m_window.pollEvent(event))
  {
    if(m_renderer.handleInput(event))
      continue;

    switch(event.type)
    {
      case sf::Event::KeyPressed:
        switch(event.key.code)
        {
          case sf::Keyboard::Space:
            this->toggleSpeed();
            break;
          case sf::Keyboard::D:
            m_renderer.DRAW_DEBUG = !m_renderer.DRAW_DEBUG;
            break;
          default:
            break;
        }
        break;
      case sf::Event::Closed:
        m_window.close();
        break;
      default:
        break;
    }
  }
}

void App::update()
{
  switch(m_speedMode)
  {
  case SpeedMode::NORMAL:
    m_elapsedtime += m_clock.restart().asSeconds();
    while(m_elapsedtime>=FIXED_DELTA_TIME)
    {
      m_elapsedtime-=FIXED_DELTA_TIME;
      m_world.update(FIXED_DELTA_TIME);
    }
    break;
  case SpeedMode::ASAP:
    while(m_clock.getElapsedTime().asSeconds() <= FRAME_TIME)
      m_world.update(FIXED_DELTA_TIME);

    m_clock.restart();
    break;
  }
}

void App::log() const
{
  std::cout << "Render Time Average:" << std::accumulate(m_renderTimes.begin(), m_renderTimes.end(), 0.0f) / m_renderTimes.size();
}

void App::render() const
{
  sf::Clock clock;

  m_renderer.begin();
  m_renderer.draw(m_world);
  m_renderer.end();

  m_window.display();
  
  std::rotate(m_renderTimes.begin(), m_renderTimes.begin()+1, m_renderTimes.end());
  m_renderTimes.back() = clock.restart().asSeconds();

  std::cout << "\033[2K\r";
  this->log();
  std::cout << "; ";
  m_world.log();
  std::cout.flush();
}

void App::toggleSpeed()
{
  m_speedMode = m_speedMode == SpeedMode::NORMAL ? SpeedMode::ASAP : SpeedMode::NORMAL;
}
