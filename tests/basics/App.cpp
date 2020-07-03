#include "App.hpp"

#include "Config.hpp"

#include <random>
#include <iostream>

App::App() : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "App"),
  m_world(sf::View({0.0, 0.0}, {WINDOW_WIDTH, WINDOW_HEIGHT}), std::random_device()()) {}

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
    if(m_world.handleInput(event))
      continue;

    switch(event.type)
    {
      case sf::Event::KeyPressed:
        switch(event.key.code)
        {
          case sf::Keyboard::Space:
            this->toggleSpeed();
            break;
          case sf::Keyboard::M:
            m_world.log();
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
    size_t i=0;
    while(m_clock.getElapsedTime().asSeconds() <= FRAME_TIME)
    {
      i++;
      m_world.update(FIXED_DELTA_TIME);
    }
    std::clog << "DEBUG: " << i << "updates in one go\n";

    m_clock.restart();
    break;
  }
}

void App::render() const
{
  m_window.clear(sf::Color::Black);
  m_window.draw(m_world);
  m_window.display();
}

void App::toggleSpeed()
{
  m_speedMode = m_speedMode == SpeedMode::NORMAL ? SpeedMode::ASAP : SpeedMode::NORMAL;
}
