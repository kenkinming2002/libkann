#pragma once

#include "Timer.hpp"

#include <SFML/Graphics/Rect.hpp>
#include <SFML/System/String.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <cmath>

inline sf::Color lerp(sf::Color a, sf::Color b, float t)
{
  return sf::Color(
    std::lerp(a.r, b.r, t),
    std::lerp(a.g, b.g, t),
    std::lerp(a.b, b.b, t)
  );
}

struct Camera
{
public:
  // Size of the render target
  sf::Vector2f renderTargetSize;

public:
  // Where it is shown
  sf::FloatRect viewport;

public:
  enum class AnchorType
  {
    CENTERED,
    TOP_LEFT
  } anchorType;

public:
  // What is shown?
  sf::Vector2f position = sf::Vector2f(0.0f, 0.0f);
  float scale = 1.0f;

public:
  void zoom(float factor) { scale *= factor; };
  void move(float x, float y) { position += scale * sf::Vector2f(x, y); };
  void reset() { *this = Camera(); }

public:
  sf::View view() const
  {
    sf::View view;

    // Where it is shown?
    view.setViewport(this->viewport);

    // What is shown?
    const sf::Vector2f size = sf::Vector2f(
      this->renderTargetSize.x * this->viewport.width,
      this->renderTargetSize.y * this->viewport.height
    );
    view.setSize(size);

    switch(anchorType)
    {
    case AnchorType::TOP_LEFT:
      view.setCenter(this->position + size / 2.0f);
      break;
    case AnchorType::CENTERED:
      view.setCenter(this->position);
      break;
    }

    view.zoom(scale);

    return view;
  };

  sf::FloatRect visibleRect() const
  {
    auto view = this->view();
    auto center = view.getCenter(), size = view.getSize();
    return sf::FloatRect(center - size/2.0f, size);
  }
};

class Renderer
{
public:
  Renderer(sf::RenderTarget& renderTarget);

public:
  static constexpr float ZOOM_SPEED = 2.0f;
  static constexpr float MOVE_SPEED = 10.0f;

  static constexpr float OUTLINE_THICKNESS = 3.0f;
  inline static const sf::Color OUTLINE_COLOR = sf::Color::Black;

  inline static const sf::Color GUI_BG_COLOR   = sf::Color::White;
  inline static const sf::Color GUI_TEXT_COLOR = sf::Color::Red;
  static constexpr unsigned GUI_TEXT_SIZE = 20;

public:
  bool handleInput(sf::Event event);

public:
  bool debug() const { return m_drawDebug; }

public:
  void begin();
  void end();

public:
  void addRectangle(sf::Vector2f position, sf::Vector2f dimension, sf::Color fillColor, float outlineThickness = OUTLINE_THICKNESS, sf::Color outlineColor = OUTLINE_COLOR);
  void addCircle(sf::Vector2f position, float radius, sf::Color fillColor, float outlineThickness = OUTLINE_THICKNESS, sf::Color outlineColor = OUTLINE_COLOR);

  void addLine(sf::Vector2f position, float length, float angle, float thickness, sf::Color fillColor);

  void addGuiText(sf::String str);

  void addBar(sf::Vector2f position, sf::Vector2f dimension, sf::Color color1, sf::Color color2, float ratio);

private:
  Camera m_mainCamera;
  Camera m_guiCamera;

private:
  sf::RenderTarget& m_renderTarget;

private:
  sf::FloatRect m_visibleRect;
  sf::VertexArray m_vertexArray;
  std::vector<sf::String> m_strs;

private:
  bool m_drawDebug = false;

private:
  Timer<> m_renderTimer;
};

