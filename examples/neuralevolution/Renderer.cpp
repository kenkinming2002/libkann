#include "Renderer.hpp"

#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/CircleShape.hpp>

#include <SFML/Graphics/Font.hpp>

#include <cmath>
#include <cassert>
#include <sstream>
#include <iomanip>

const sf::Color Renderer::GUI_TEXT_COLOR = sf::Color::Red;

namespace
{
  sf::Vector2f convert(Eigen::Vector2d vec)
  {
    return sf::Vector2f(static_cast<float>(vec(0)), static_cast<float>(vec(1)));
  }

  sf::Color lerp(sf::Color a, sf::Color b, float t)
  {
    return sf::Color(
      std::lerp(a.r, b.r, t),
      std::lerp(a.g, b.g, t),
      std::lerp(a.b, b.b, t)
    );
  }

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
  : m_renderTarget(renderTarget), m_vertexArray(sf::PrimitiveType::Triangles),
    m_renderTimes{} {}

bool Renderer::handleInput(sf::Event event)
{
  switch(event.type) {
    case sf::Event::KeyPressed:
      switch(event.key.code)
      {
        // Zoom in / Zoom out
        case sf::Keyboard::Add:
          m_camera.zoom(1.0f / ZOOM_SPEED);
          return true;
        case sf::Keyboard::Subtract:
          m_camera.zoom(ZOOM_SPEED);
          return true;

        // Up/Down/Left/Right
        case sf::Keyboard::H:
        case sf::Keyboard::Left:
          m_camera.move(-MOVE_SPEED, 0.0f);
          return true;
        case sf::Keyboard::J:
        case sf::Keyboard::Down:
          m_camera.move(0.0f, MOVE_SPEED);
          return true;
        case sf::Keyboard::K:
        case sf::Keyboard::Up:
          m_camera.move(0.0f, -MOVE_SPEED);
          return true;
        case sf::Keyboard::L:
        case sf::Keyboard::Right:
          m_camera.move(MOVE_SPEED, 0.0f);
          return true;

          // Reset
        case sf::Keyboard::Equal:
          m_camera.reset();
          return true;

        case sf::Keyboard::D:
          m_drawDebug = !m_drawDebug;
          return true;
        default:
          return false;
      }
    // Zoom in / Zoom out
    case sf::Event::MouseWheelScrolled:
    {
      float factor = event.mouseWheelScroll.delta >= 0.0f ? event.mouseWheelScroll.delta * ZOOM_SPEED : -1.0f / (event.mouseWheelScroll.delta * ZOOM_SPEED);
      assert(factor > 0.0f);
      m_camera.zoom(factor);
      return true;
    }
    default:
      return false;
  }
}

void Renderer:: begin()
{
  m_renderTimeClock.restart();

  m_vertexArray.clear();
  m_strs.clear();

  m_renderTarget.clear(sf::Color::Black);
}

void Renderer:: end()
{
  this->addGuiText(concatenate("Render Time:",
    std::accumulate(m_renderTimes.begin(), m_renderTimes.end(), 0.0f) / m_renderTimes.size())
  );

  // VertexArray
  m_renderTarget.setView(m_camera.view(m_renderTarget));
  m_renderTarget.draw(m_vertexArray);

  // Gui text
  m_renderTarget.setView(this->guiView());

  static sf::Font font = [](){
    sf::Font font;
    if(!font.loadFromFile("resources/fonts/arial.ttf"))
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

  // Render time measurement
  std::rotate(m_renderTimes.begin(), m_renderTimes.begin()+1, m_renderTimes.end());
  m_renderTimes.back() = m_renderTimeClock.getElapsedTime().asSeconds();
}

void Renderer::draw(const Creature& creature)
{
  float radius = CONFIG.creature.radius * creature.m_health / CONFIG.creature.maxHealth;
  sf::Color color = lerp(sf::Color::Yellow, sf::Color::Green, creature.m_energy / CONFIG.creature.maxEnergy);
  this->addCircle(convert(creature.m_position), radius, color, OUTLINE_THICKNESS, sf::Color::Black);

  if(m_drawDebug)
  {
    for(const auto& eye: creature.m_eyes)
    {
      float angle = creature.m_rotation+eye.angle;
      this->addLine(convert(creature.m_position), eye.distance, angle, 3.0f, sf::Color::Red);
    }
  }
}

void Renderer::draw(const BerryBush& berryBush)
{
  const float BAR_THICKNESS       = CONFIG.berryBush.radius * 0.2f;
  const float BAR_VERTICAL_OFFSET = CONFIG.berryBush.radius * 1.2f;

  this->addCircle(convert(berryBush.m_position), CONFIG.berryBush.radius, sf::Color::Green, OUTLINE_THICKNESS, sf::Color::Black);
  this->addBar(
    convert(berryBush.m_position) + sf::Vector2f(0.0f, BAR_VERTICAL_OFFSET), 
    sf::Vector2f(2.0f * CONFIG.berryBush.radius, BAR_THICKNESS), 
    sf::Color::Green, sf::Color::Red, 
    static_cast<float>(berryBush.m_berryCount) / CONFIG.berryBush.maxBerryCount
  );
}

void Renderer::draw(const World& world)
{
  auto info = world.info();

  this->addGuiText(concatenate("Creatures:", info.healthyCreaturesCount, "/", info.creaturesCount, "(Healthy/All)"));
  this->addGuiText(concatenate("Statistics:", info.deathToll, "/", info.birthCount, "(DeathToll/BirthCount)"));
  this->addGuiText(concatenate("Time:", info.realTime, "/", info.worldTime, "/", info.worldTime / info.realTime, 
        "(Real/World/Ratio)"));

  this->addRectangle(sf::Vector2f(0.0f, 0.0f), convert(world.m_dimension), sf::Color::White);

  for(std::reference_wrapper<const BerryBush> berryBush: world.m_berryBushes.all())
    this->draw(berryBush);

  for(std::reference_wrapper<const Creature> creature: world.m_creatures.all())
    this->draw(creature);
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

sf::View Renderer::guiView() const
{
  sf::Vector2f windowSize(m_renderTarget.getSize()); 
  return sf::View(windowSize/2.0f, windowSize); 
}


