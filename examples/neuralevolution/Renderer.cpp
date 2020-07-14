#include "Renderer.hpp"

#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/CircleShape.hpp>

#include <SFML/Graphics/Font.hpp>

#include <cmath>

static sf::Vector2f convert(Eigen::Vector2d vec)
{
  return {vec(0), vec(1)};
}

static sf::Color lerp(sf::Color a, sf::Color b, float t)
{
  return sf::Color(
    std::lerp(a.r, b.r, t),
    std::lerp(a.g, b.g, t),
    std::lerp(a.b, b.b, t)
  );
}

Renderer::Renderer(sf::RenderTarget& renderTarget, sf::View view)
  : m_renderTarget(renderTarget), m_vertexArray(sf::PrimitiveType::Triangles),
    m_defaultView(view), m_view(view) {}

bool Renderer::handleInput(sf::Event event)
{
  auto zoom = [this](float amount){
    m_view.zoom(amount);
    m_scale *= amount;
    return true;
  };
  auto move = [this](float x, float y){
    m_view.move(x * m_scale, y * m_scale);
    return true;
  };
  auto reset = [this](){
    m_view = m_defaultView;
    return true;
  };

  switch(event.type) {
    case sf::Event::KeyPressed:
      switch(event.key.code)
      {
        // Zoom in / Zoom out
        case sf::Keyboard::Add:
          return zoom(1.0f / ZOOM_SPEED);
        case sf::Keyboard::Subtract:
          return zoom(ZOOM_SPEED);

          // Up/Down/Left/Right
        case sf::Keyboard::H:
        case sf::Keyboard::Left:
          return move(-MOVE_SPEED, 0.0f);
        case sf::Keyboard::J:
        case sf::Keyboard::Down:
          return move(0.0f, MOVE_SPEED);
        case sf::Keyboard::K:
        case sf::Keyboard::Up:
          return move(0.0f, -MOVE_SPEED);
        case sf::Keyboard::L:
        case sf::Keyboard::Right:
          return move(MOVE_SPEED, 0.0f);

          // Reset
        case sf::Keyboard::Equal:
          return reset();
        default:
          return false;
      }
    // Zoom in / Zoom out
    case sf::Event::MouseWheelScrolled:
      if(event.mouseWheelScroll.delta >= 0.0f)
        return zoom(event.mouseWheelScroll.delta * ZOOM_SPEED);
      else
        return zoom(1.0f / (event.mouseWheelScroll.delta * ZOOM_SPEED));
      break;
    default:
      return false;
  }
}

void Renderer:: begin()
{
  m_vertexArray.clear();
  m_texts.clear();

  m_renderTarget.clear(sf::Color::Black);
  m_renderTarget.setView(m_view);
}

void Renderer:: end()
{
  m_renderTarget.draw(m_vertexArray);
  for(const auto& text: m_texts)
    m_renderTarget.draw(text);
}

void Renderer::draw(const Creature& creature)
{
  float radius = CONFIG.creature.radius * creature.m_health / CONFIG.creature.maxHealth;
  sf::Color color = lerp(sf::Color::Yellow, sf::Color::Green, creature.m_energy / CONFIG.creature.maxEnergy);
  this->addCircle(convert(creature.m_position), radius, color, sf::Color::Black);

  if(DRAW_DEBUG)
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
  this->addCircle(convert(berryBush.m_position), CONFIG.berryBush.radius, sf::Color::Green, sf::Color::Black);

  // Consider using a multiplier
  const auto textSize = static_cast<unsigned>(CONFIG.berryBush.radius);
  this->addText(std::to_string(berryBush.m_berryCount), convert(berryBush.m_position), textSize);
}

void Renderer::draw(const World& world)
{
  sf::RectangleShape rectangleShape;

  rectangleShape.setPosition(0.0f, 0.0f);
  rectangleShape.setSize({static_cast<float>(world.m_dimension(0)), static_cast<float>(world.m_dimension(1))});
  rectangleShape.setOrigin(world.m_dimension(0)/2.0, world.m_dimension(1)/2.0);

  rectangleShape.setFillColor(sf::Color::White);

  m_renderTarget.draw(rectangleShape);

  for(std::reference_wrapper<const BerryBush> berryBush: world.m_berryBushes.all())
    this->draw(berryBush);

  for(std::reference_wrapper<const Creature> creature: world.m_creatures.all())
    this->draw(creature);
}

void Renderer::addCircle(sf::Vector2f position, float radius, sf::Color fillColor, sf::Color outlineColor)
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
  sf::Vertex points[POINT_COUNT];

  center = sf::Vertex(position, fillColor);
  for(size_t i=0; i<POINT_COUNT; ++i)
  {
    sf::Vector2f pointPosition = position + radius * unitVectors[i];
    points[i] = sf::Vertex(pointPosition, fillColor);
  }

  for(size_t i=0; i<POINT_COUNT; ++i)
  {
    m_vertexArray.append(center);
    m_vertexArray.append(points[i]);
    m_vertexArray.append(points[i+1 != POINT_COUNT ? i+1 : 0]);
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

void Renderer::addText(const sf::String& str, sf::Vector2f position, unsigned characterSize)
{
  // TODO: Draw the berries
  static sf::Font font = [](){
    sf::Font font;
    if(!font.loadFromFile("resources/fonts/arial.ttf"))
      throw std::runtime_error("Failed to load font");

    return font;
  }();

  m_texts.emplace_back(str, font, characterSize);
  m_texts.back().setPosition(position);
  m_texts.back().setFillColor(sf::Color::Black);

  auto textRect = m_texts.back().getLocalBounds();
  m_texts.back().setOrigin(textRect.left + textRect.width/2.0f, textRect.top  + textRect.height/2.0f);
}



