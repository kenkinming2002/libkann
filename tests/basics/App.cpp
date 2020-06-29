#include "App.hpp"

#include <random>
#include <iostream>

App::App() : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "App"),
  m_world(50, 10, {WORLD_WIDTH, WORLD_HEIGHT}, std::random_device()())
{
  m_window.setView(sf::View({0.0f, 0.0f}, {WORLD_WIDTH, WORLD_HEIGHT}));
}

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
    switch(event.type)
    {
    case sf::Event::KeyPressed:
      switch(event.key.code)
      {
      case sf::Keyboard::Space:
        this->toggleSpeed();
        break;
      case sf::Keyboard::Add:
        this->zoom(1.0f / ZOOM_SPEED);
        break;
      case sf::Keyboard::Subtract:
        this->zoom(ZOOM_SPEED);
        break;

      case sf::Keyboard::H:
      case sf::Keyboard::Left:
        this->move(-MOVE_SPEED, 0.0f);
        break;
      case sf::Keyboard::J:
      case sf::Keyboard::Down:
        this->move(0.0f, MOVE_SPEED);
        break;
      case sf::Keyboard::K:
      case sf::Keyboard::Up:
        this->move(0.0f, -MOVE_SPEED);
        break;
      case sf::Keyboard::L:
      case sf::Keyboard::Right:
        this->move(MOVE_SPEED, 0.0f);
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
  size_t i=0;

  m_elapsedtime += m_clock.restart().asSeconds();
  if(m_elapsedtime > FIXED_DELTA_TIME * MAX_FIXED_UPDATE_PER_FRAME)
  {
    std::clog << "Warning: Large total elapsed time - clamping to prevent death spiral\n";
    m_elapsedtime = FIXED_DELTA_TIME * MAX_FIXED_UPDATE_PER_FRAME;
  }

  while(m_elapsedtime>=FIXED_DELTA_TIME / m_speedUp)
  {
    m_elapsedtime-=FIXED_DELTA_TIME / m_speedUp;
    m_world.update(FIXED_DELTA_TIME);
    ++i;
  }
}

void App::render() const
{
  m_window.clear(sf::Color::White);
  m_window.draw(m_world);
  m_window.display();
}

void App::toggleSpeed()
{
  m_superSpeed = !m_superSpeed;
  m_speedUp = m_superSpeed ? SUPER_SPEED_SPEED_UP : NORMAL_SPEED_UP;
}

void App::zoom(float scale)
{
  auto view = m_window.getView();
  view.zoom(scale);
  m_window.setView(view);
}

void App::move(float x, float y)
{
  auto view = m_window.getView();
  view.move(x, y);
  m_window.setView(view);
}
