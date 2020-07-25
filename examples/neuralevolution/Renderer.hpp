#pragma once

#include "Creature.hpp"
#include "BerryBush.hpp"
#include "World.hpp"
#include "Timer.hpp"

#include <SFML/System/String.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/VertexArray.hpp>

class Renderer
{
private:
  struct Camera
  {
  public:
    Camera() : center(0.0f, 0.0f), scale(1.0f) {}

  public:
    void zoom(float factor) { scale *= factor; };
    void move(float x, float y) { center += scale * sf::Vector2f(x, y); };
    void reset() { *this = Camera(); }

  public:
    sf::View view(const sf::RenderTarget& renderTarget) const
    {
      sf::View view(center, sf::Vector2f(renderTarget.getSize()));
      view.zoom(scale);
      return view;
    };

    sf::FloatRect visibleRect(const sf::RenderTarget& renderTarget) const
    {
      auto view = this->view(renderTarget);
      auto center = view.getCenter(), size = view.getSize();
      return sf::FloatRect(center - size/2.0f, size);
    }

  public:
    sf::Vector2f center;
    float scale;
  };

public:
  Renderer(sf::RenderTarget& renderTarget);

public:
  static constexpr float ZOOM_SPEED = 2.0f;
  static constexpr float MOVE_SPEED = 10.0f;

  static constexpr float OUTLINE_THICKNESS = 3.0f;
  static const sf::Color GUI_TEXT_COLOR /*= sf::Color::Red*/;
  static constexpr unsigned GUI_TEXT_SIZE = 30;

public:
  bool handleInput(sf::Event event, World& world);

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

  void addGuiText(sf::String str);

  void addBar(sf::Vector2f position, sf::Vector2f dimension, sf::Color color1, sf::Color color2, float ratio);

private:
  sf::View guiView() const;

private:
  Camera m_camera;
  sf::FloatRect m_visibleRect;

private:
  sf::RenderTarget& m_renderTarget;

private:
  sf::VertexArray m_vertexArray;
  std::vector<sf::String> m_strs;

private:
  bool m_drawDebug = false;

private:
  Timer<> m_renderTimer;
};

