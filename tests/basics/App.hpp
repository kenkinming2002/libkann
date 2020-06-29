#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/System/Clock.hpp>

#include "World.hpp"

class App
{
public:
  static constexpr unsigned WINDOW_WIDTH = 1000;
  static constexpr unsigned WINDOW_HEIGHT = 1000;

  static constexpr unsigned WORLD_WIDTH = 1000;
  static constexpr unsigned WORLD_HEIGHT = 1000;

public:
  static constexpr float NORMAL_SPEED_UP = 1.0f;
  static constexpr float SUPER_SPEED_SPEED_UP = 5.0f;

public:
  static constexpr float FIXED_DELTA_TIME = 1 / 30.0f;
  static constexpr size_t MAX_FIXED_UPDATE_PER_FRAME = 10;

public:
  static constexpr float ZOOM_SPEED = 2.0f;
  static constexpr float MOVE_SPEED = 10.0f;

public:
  App();

public:
  void run();

private:
  bool loop();

  void handleInput();
  void update();
  void render() const;

private:
  void toggleSpeed();
  void zoom(float scale);
  void move(float x, float y);

private:
  mutable sf::RenderWindow m_window;

public:
  sf::Clock m_clock;
  float m_elapsedtime = 0.0f;
  bool m_superSpeed;
  float m_speedUp = NORMAL_SPEED_UP;

private:
  World m_world;
};
