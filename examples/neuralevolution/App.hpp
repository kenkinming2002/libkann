#pragma once

#include <array>

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/System/Clock.hpp>

#include "Renderer.hpp"
#include "World.hpp"

class App
{
public:
  using random_engine_type = World::random_engine_type;
  using seed_type = World::seed_type;

public:
  static constexpr unsigned WINDOW_WIDTH = 1000;
  static constexpr unsigned WINDOW_HEIGHT = 1000;

public:
  static constexpr float FIXED_DELTA_TIME = 1 / 30.0f;
  static constexpr float FRAME_TIME = 1 / 60.0f;

public:
  App(seed_type seed);

public:
  void run();

private:
  bool loop();

  void handleInput();
  void update();
  void render() const;

private:
  void toggleSpeed();

private:
  sf::Clock m_clock;
  float m_elapsedtime = 0.0f;
  enum class SpeedMode
  {
    NORMAL,
    ASAP
  } m_speedMode = SpeedMode::NORMAL;

private:
  World m_world;

private:
  mutable sf::RenderWindow m_window;
  mutable Renderer m_renderer;
};
