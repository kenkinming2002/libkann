#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"
#include "World.hpp"

#include <SFML/Graphics/View.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/RenderTarget.hpp>

class Renderer
{
public:
  Renderer(sf::RenderTarget& renderTarget, sf::View view)
    : m_renderTarget(renderTarget), m_defaultView(view), m_view(view) {}

public:
  static constexpr float ZOOM_SPEED = 2.0f;
  static constexpr float MOVE_SPEED = 10.0f;

public:
  bool handleInput(sf::Event event);

public:
  void begin();
  void end();

public:
  void draw(const Creature& creature);
  void draw(const BerryBush& berryBush);
  void draw(const World& world);

private:
  sf::RenderTarget& m_renderTarget;

private:
  const sf::View m_defaultView;
  sf::View m_view;
  float m_scale = 1.0f;

public:
  bool DRAW_DEBUG = false;
};
