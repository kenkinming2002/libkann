#include "Renderer.hpp"

#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/CircleShape.hpp>

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>

static sf::Color lerp(sf::Color a, sf::Color b, float t)
{
  return sf::Color(
    std::lerp(a.r, b.r, t),
    std::lerp(a.g, b.g, t),
    std::lerp(a.b, b.b, t)
  );
}

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
  m_renderTarget.clear(sf::Color::Black);
  m_renderTarget.setView(m_view);
}

void Renderer:: end()
{
}

void Renderer::draw(const Creature& creature)
{
  sf::CircleShape circleShape;
  {
    float radius = CONFIG.creature.radius * creature.m_health / CONFIG.creature.maxHealth;

    circleShape.setRadius(radius);
    circleShape.setOrigin({radius, radius});

    circleShape.setFillColor(lerp(sf::Color::Yellow, sf::Color::Green, creature.m_energy / CONFIG.creature.maxEnergy));

    circleShape.setOutlineThickness(2);
    circleShape.setOutlineColor(sf::Color::Black);
  }
  circleShape.setPosition(creature.m_position(0), creature.m_position(1));
  m_renderTarget.draw(circleShape);

  if(DRAW_DEBUG)
  {
    static constexpr float THICKNESS = 3.0f;

    for(const auto& eye: creature.m_eyes)
    {
      sf::RectangleShape rectangleShape;
      {
        rectangleShape.setSize({static_cast<float>(eye.distance), THICKNESS});
        rectangleShape.setOrigin(0.0f, THICKNESS/2.0f);

        rectangleShape.setFillColor(sf::Color::Red);
        rectangleShape.setRotation((creature.m_rotation+eye.angle) * 360.0 / (2*M_PI));
      }
      rectangleShape.setPosition(creature.m_position(0), creature.m_position(1));
      m_renderTarget.draw(rectangleShape);
    }
  }
}

void Renderer::draw(const BerryBush& berryBush)
{
  // Consider using a multiplier
  const auto TEXT_SIZE = static_cast<unsigned>(CONFIG.berryBush.radius);

  sf::CircleShape circleShape;

  // Draw the bush
  circleShape.setRadius(static_cast<float>(CONFIG.berryBush.radius));
  circleShape.setOrigin({static_cast<float>(CONFIG.berryBush.radius), static_cast<float>(CONFIG.berryBush.radius)});

  circleShape.setFillColor(sf::Color::Green);

  circleShape.setOutlineThickness(2);
  circleShape.setOutlineColor(sf::Color::Black);

  circleShape.setPosition(berryBush.m_position(0), berryBush.m_position(1));
  m_renderTarget.draw(circleShape);

  // TODO: Draw the berries
  static sf::Font font = [](){
    sf::Font font;
    if(!font.loadFromFile("resources/fonts/arial.ttf"))
      throw std::runtime_error("Failed to load font");

    return font;
  }();

  sf::Text text;

  text.setFont(font);
  text.setCharacterSize(TEXT_SIZE);
  text.setFillColor(sf::Color::Black);

  text.setString(std::to_string(berryBush.m_berryCount));

  auto textRect = text.getLocalBounds();
  text.setOrigin(textRect.left + textRect.width/2.0f,
                 textRect.top  + textRect.height/2.0f);
  text.setPosition(berryBush.m_position(0), berryBush.m_position(1));

  m_renderTarget.draw(text);
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

