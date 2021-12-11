#include "Renderer.hpp"

#include "Config.hpp"

#include <SFML/Graphics/Rect.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/Font.hpp>

#include <SFML/System/Vector2.hpp>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <sstream>
#include <iomanip>

namespace
{
  template<typename... Args>
  std::string concatenate(const Args&... args)
  {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);
    (ss << ... << args);
    return ss.str();
  }
}

Renderer::Renderer(sf::RenderTarget& renderTarget)
  : m_renderTarget(renderTarget), m_vertexArray(sf::PrimitiveType::Triangles)
{
  m_mainCamera = Camera{
    .renderTargetSize = sf::Vector2f(renderTarget.getSize()),
    .viewport = sf::FloatRect(0.0f, 0.15f, 1.0f, 0.85f),
    .anchorType = Camera::AnchorType::CENTERED,
    .position = sf::Vector2f(0.0f, 0.0f),
    .scale = 1.0f
  };

  m_guiCamera = Camera{
    .renderTargetSize = sf::Vector2f(renderTarget.getSize()),
    .viewport = sf::FloatRect(0.0f, 0.0f, 1.0f, 0.15f),
    .anchorType = Camera::AnchorType::TOP_LEFT,
    .position = sf::Vector2f(0.0f, 0.0f),
    .scale = 1.0f
  };
}

bool Renderer::handleInput(sf::Event event)
{
  switch(event.type) {
    case sf::Event::KeyPressed:
      switch(event.key.code)
      {
        // Zoom in / Zoom out
        case sf::Keyboard::Add:
          m_mainCamera.zoom(1.0f / ZOOM_SPEED);
          return true;
        case sf::Keyboard::Subtract:
          m_mainCamera.zoom(ZOOM_SPEED);
          return true;

        // Up/Down/Left/Right
        case sf::Keyboard::H:
        case sf::Keyboard::Left:
          m_mainCamera.move(-MOVE_SPEED, 0.0f);
          return true;
        case sf::Keyboard::J:
        case sf::Keyboard::Down:
          m_mainCamera.move(0.0f, MOVE_SPEED);
          return true;
        case sf::Keyboard::K:
        case sf::Keyboard::Up:
          m_mainCamera.move(0.0f, -MOVE_SPEED);
          return true;
        case sf::Keyboard::L:
        case sf::Keyboard::Right:
          m_mainCamera.move(MOVE_SPEED, 0.0f);
          return true;

          // Reset
        case sf::Keyboard::Equal:
          m_mainCamera.reset();
          return true;

        case sf::Keyboard::D:
          m_drawDebug = !m_drawDebug;
          return true;
        default:
          return false;
      }
    // Window resize
    case sf::Event::Resized:
    {
      m_mainCamera.renderTargetSize = sf::Vector2f(event.size.width, event.size.height);
      m_guiCamera.renderTargetSize  = sf::Vector2f(event.size.width, event.size.height);
      return true;
    }
    // Zoom in / Zoom out
    case sf::Event::MouseWheelScrolled:
    {
      float factor = event.mouseWheelScroll.delta >= 0.0f ? event.mouseWheelScroll.delta * ZOOM_SPEED : -1.0f / (event.mouseWheelScroll.delta * ZOOM_SPEED);
      assert(factor > 0.0f);
      m_mainCamera.zoom(factor);
      return true;
    }
    default:
      return false;
  }
}

void Renderer:: begin()
{
  m_renderTimer.begin();

  m_vertexArray.clear();
  m_strs.clear();

  m_visibleRect = m_mainCamera.visibleRect();
}

void Renderer::end()
{
  this->addGuiText(concatenate("Render Time:", m_renderTimer.average()));

  // Drawing
  m_renderTarget.clear(sf::Color::Black);

  // Main Camera
  {
    m_renderTarget.setView(m_mainCamera.view());

    // VertexArray
    m_renderTarget.draw(m_vertexArray);
  }

  // Gui Camera
  {
    m_renderTarget.setView(m_guiCamera.view());

    // Background
    sf::FloatRect visibleRect = m_guiCamera.visibleRect();
    sf::RectangleShape rectangleShape;
    rectangleShape.setFillColor(GUI_BG_COLOR);
    rectangleShape.setPosition(sf::Vector2f(visibleRect.left, visibleRect.top));
    rectangleShape.setSize(sf::Vector2f(visibleRect.width, visibleRect.height));
    m_renderTarget.draw(rectangleShape);

    // Gui text
    static sf::Font font = [](){
      sf::Font font;
      if(!font.loadFromFile("resources/fonts/NotoSansMono-Regular.ttf"))
        throw std::runtime_error("Failed to load font");

      return font;
    }();

    sf::Text text;
    text.setFont(font);
    text.setFillColor(GUI_TEXT_COLOR);
    text.setCharacterSize(GUI_TEXT_SIZE);

    sf::Vector2f position(0.0f, 0.0f);
    for(const sf::String& str: m_strs)
    {
      text.setPosition(position);
      text.setString(str);
      m_renderTarget.draw(text);
      position.y += text.getLocalBounds().height;
    }
  }

  m_renderTimer.end();
}

void Renderer::addRectangle(sf::Vector2f position, sf::Vector2f dimension, sf::Color fillColor, float outlineThickness, sf::Color outlineColor)
{
  sf::Vector2f pointsInner[4];
  sf::Vector2f pointsOuter[4];

  pointsInner[0] = position + sf::Vector2f(-dimension.x/2.0f, -dimension.y/2.0f);
  pointsInner[1] = position + sf::Vector2f(-dimension.x/2.0f,  dimension.y/2.0f);
  pointsInner[2] = position + sf::Vector2f( dimension.x/2.0f,  dimension.y/2.0f);
  pointsInner[3] = position + sf::Vector2f( dimension.x/2.0f, -dimension.y/2.0f);

  pointsOuter[0] = position + sf::Vector2f(-dimension.x/2.0f-outlineThickness, -dimension.y/2.0f-outlineThickness);
  pointsOuter[1] = position + sf::Vector2f(-dimension.x/2.0f-outlineThickness,  dimension.y/2.0f+outlineThickness);
  pointsOuter[2] = position + sf::Vector2f( dimension.x/2.0f+outlineThickness,  dimension.y/2.0f+outlineThickness);
  pointsOuter[3] = position + sf::Vector2f( dimension.x/2.0f+outlineThickness, -dimension.y/2.0f-outlineThickness);

  if(fillColor != sf::Color::Transparent)
  {
    m_vertexArray.append(sf::Vertex(pointsInner[0], fillColor));
    m_vertexArray.append(sf::Vertex(pointsInner[1], fillColor));
    m_vertexArray.append(sf::Vertex(pointsInner[2], fillColor));

    m_vertexArray.append(sf::Vertex(pointsInner[2], fillColor));
    m_vertexArray.append(sf::Vertex(pointsInner[3], fillColor));
    m_vertexArray.append(sf::Vertex(pointsInner[0], fillColor));
  }

  if(outlineThickness != 0.0f && outlineColor != sf::Color::Transparent)
  {
    // Outline
    for(size_t i=0; i<4; ++i)
    {
      m_vertexArray.append(sf::Vertex(pointsInner[i], outlineColor));
      m_vertexArray.append(sf::Vertex(pointsOuter[i], outlineColor));
      m_vertexArray.append(sf::Vertex(pointsOuter[i+1!=4 ? i+1 : 0], outlineColor));

      m_vertexArray.append(sf::Vertex(pointsOuter[i+1!=4 ? i+1 : 0], outlineColor));
      m_vertexArray.append(sf::Vertex(pointsInner[i+1!=4 ? i+1 : 0], outlineColor));
      m_vertexArray.append(sf::Vertex(pointsInner[i], outlineColor));
    }
  }
}

void Renderer::addCircle(sf::Vector2f position, float radius, sf::Color fillColor, float outlineThickness, sf::Color outlineColor)
{
  // Occlusion culling
  auto boundingRectCenter = position;
  auto boundingRectHalfSize = sf::Vector2f(radius + outlineThickness, radius + outlineThickness);

  auto boundingRect = sf::FloatRect(boundingRectCenter - boundingRectHalfSize, 2.0f * boundingRectHalfSize);
  if(!m_visibleRect.intersects(boundingRect))
    return;

  static constexpr size_t POINT_COUNT = 30;
  static auto unitVectors = [](){
    std::array<sf::Vector2f, POINT_COUNT> result;
    for(size_t i=0; i<POINT_COUNT; ++i)
    {
      float angle = (2.0f * M_PI * i) / POINT_COUNT;
      result[i] = sf::Vector2f(std::cos(angle), std::sin(angle));
    }
    return result;
  }();

  sf::Vertex center;
  sf::Vector2f pointsInner[POINT_COUNT];
  sf::Vector2f pointsOuter[POINT_COUNT];

  center = sf::Vertex(position, fillColor);
  for(size_t i=0; i<POINT_COUNT; ++i)
    pointsInner[i] = position + radius * unitVectors[i];
  for(size_t i=0; i<POINT_COUNT; ++i)
    pointsOuter[i] = position + (radius+outlineThickness) * unitVectors[i];

  // Innner triangle fan
  for(size_t i=0; i<POINT_COUNT; ++i)
  {
    m_vertexArray.append(center);
    m_vertexArray.append(sf::Vertex(pointsInner[i], fillColor));
    m_vertexArray.append(sf::Vertex(pointsInner[i+1 != POINT_COUNT ? i+1 : 0], fillColor));
  }

  // Outline
  for(size_t i=0; i<POINT_COUNT; ++i)
  {
    m_vertexArray.append(sf::Vertex(pointsInner[i], outlineColor));
    m_vertexArray.append(sf::Vertex(pointsOuter[i], outlineColor));
    m_vertexArray.append(sf::Vertex(pointsOuter[i+1!=POINT_COUNT ? i+1 : 0], outlineColor));

    m_vertexArray.append(sf::Vertex(pointsOuter[i+1!=POINT_COUNT ? i+1 : 0], outlineColor));
    m_vertexArray.append(sf::Vertex(pointsInner[i+1!=POINT_COUNT ? i+1 : 0], outlineColor));
    m_vertexArray.append(sf::Vertex(pointsInner[i], outlineColor));
  }
}

void Renderer::addLine(sf::Vector2f position, float length, float angle, float thickness, sf::Color fillColor)
{
  sf::Vector2f forward = length * sf::Vector2f(std::cos(angle), std::sin(angle));
  sf::Vector2f right = (thickness / 2.0f) * sf::Vector2f(-std::sin(angle), std::cos(angle));

  // Note: This bounding rect is actually slightly off since it disregard the
  //       thickness of the line, but it should not matter, since it is
  //       pointness to render tiny part if the line poking out.
  auto boundingRect = sf::FloatRect(position, forward);
  if(!m_visibleRect.intersects(boundingRect))
    return;

  sf::Vertex points[4];

  points[0] = sf::Vertex(position           + right, fillColor);
  points[1] = sf::Vertex(position + forward + right, fillColor);
  points[2] = sf::Vertex(position + forward - right, fillColor);
  points[3] = sf::Vertex(position           - right, fillColor);

  m_vertexArray.append(points[0]);
  m_vertexArray.append(points[1]);
  m_vertexArray.append(points[2]);

  m_vertexArray.append(points[2]);
  m_vertexArray.append(points[3]);
  m_vertexArray.append(points[0]);
}

void Renderer::addGuiText(sf::String str)
{
  m_strs.push_back(std::move(str));
}

void Renderer::addBar(sf::Vector2f position, sf::Vector2f dimension, sf::Color color1, sf::Color color2, float ratio)
{
  auto boundingRect = sf::FloatRect(position - dimension/2.0f, dimension);
  if(!m_visibleRect.intersects(boundingRect))
    return;

  sf::Vertex points[8];

  points[0] = sf::Vertex(position + sf::Vector2f(-dimension.x/2.0f, -dimension.y/2.0f), color1);
  points[1] = sf::Vertex(position + sf::Vector2f(-dimension.x/2.0f,  dimension.y/2.0f), color1);

  points[2] = sf::Vertex(position + sf::Vector2f(-dimension.x/2.0f + ratio*dimension.x,  dimension.y/2.0f), color1);
  points[3] = sf::Vertex(position + sf::Vector2f(-dimension.x/2.0f + ratio*dimension.x, -dimension.y/2.0f), color1);

  points[4] = sf::Vertex(position + sf::Vector2f(-dimension.x/2.0f + ratio*dimension.x, -dimension.y/2.0f), color2);
  points[5] = sf::Vertex(position + sf::Vector2f(-dimension.x/2.0f + ratio*dimension.x,  dimension.y/2.0f), color2);

  points[6] = sf::Vertex(position + sf::Vector2f( dimension.x/2.0f,  dimension.y/2.0f), color2);
  points[7] = sf::Vertex(position + sf::Vector2f( dimension.x/2.0f, -dimension.y/2.0f), color2);

  m_vertexArray.append(points[0]);
  m_vertexArray.append(points[1]);
  m_vertexArray.append(points[2]);

  m_vertexArray.append(points[2]);
  m_vertexArray.append(points[3]);
  m_vertexArray.append(points[0]);

  m_vertexArray.append(points[4]);
  m_vertexArray.append(points[5]);
  m_vertexArray.append(points[6]);

  m_vertexArray.append(points[6]);
  m_vertexArray.append(points[7]);
  m_vertexArray.append(points[4]);
}

