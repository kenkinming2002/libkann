#include "App.hpp"

#include "Config.hpp"

#include <SFML/Window/Event.hpp>

App::App(seed_type seed) 
  : m_world(seed), 
    m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "App"), 
    m_renderer(m_window) {}

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
    if(m_renderer.handleInput(event, m_world))
      continue;

    switch(event.type)
    {
      case sf::Event::KeyPressed:
        switch(event.key.code)
        {
          case sf::Keyboard::Space:
            this->toggleSpeed();
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

void App::render() const
{
  m_renderer.begin();
  m_renderer.draw(m_world);
  m_renderer.end();

  m_window.display();
}

void App::toggleSpeed()
{
  m_speedMode = m_speedMode == SpeedMode::NORMAL ? SpeedMode::ASAP : SpeedMode::NORMAL;
}
