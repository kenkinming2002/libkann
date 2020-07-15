#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"
#include "World.hpp"

#include <SFML/Graphics/View.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/Text.hpp>

class Renderer
{
public:
  Renderer(sf::RenderTarget& renderTarget, sf::View view);

public:
  static constexpr float ZOOM_SPEED = 2.0f;
  static constexpr float MOVE_SPEED = 10.0f;

  static constexpr float OUTLINE_THICKNESS = 3.0f;

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
  void addRectangle(sf::Vector2f position, sf::Vector2f dimension, sf::Color fillColor, float outlineThickness = 0.0f, sf::Color outlineColor = sf::Color::Transparent);
  void addCircle(sf::Vector2f position, float radius, sf::Color fillColor, float outlineThickness = 0.0f, sf::Color outlineColor = sf::Color::Transparent);

  void addLine(sf::Vector2f position, float length, float angle, float thickness, sf::Color fillColor);

  void addText(const sf::String& str, sf::Vector2f position, unsigned characterSize);

  void addBar(sf::Vector2f position, sf::Vector2f dimension, sf::Color color1, sf::Color color2, float ratio);

private:
  sf::RenderTarget& m_renderTarget;

private:
  sf::VertexArray m_vertexArray;
  std::vector<sf::Text> m_texts;

private:
  const sf::View m_defaultView;
  sf::View m_view;
  float m_scale = 1.0f;

public:
  bool DRAW_DEBUG = false;
};
