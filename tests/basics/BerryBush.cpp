#include "BerryBush.hpp"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/RenderTarget.hpp>

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>

#include <iostream>
#include <cmath>

#include "Config.hpp"

BerryBush::BerryBush(Eigen::Vector2d position) 
  : m_position(position), m_berryCount(CONFIG.berryBush.maxBerryCount), m_growth(0) {}

void BerryBush::update(float dt)
{
  m_growth += dt * CONFIG.berryBush.growthRate;;
  if(m_growth>=1.0f)
  {
    m_growth-=1.0f;
    if(m_berryCount != CONFIG.berryBush.maxBerryCount)
      ++m_berryCount;
  }
}

void BerryBush::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
  static constexpr float BUSH_RADIUS = 30.0f;
  static constexpr float BERRY_RADIUS = 10.0f;
  static constexpr unsigned TEXT_SIZE = 30;

  sf::CircleShape circleShape;

  // Draw the bush
  circleShape.setRadius(BUSH_RADIUS);
  circleShape.setOrigin({BUSH_RADIUS, BUSH_RADIUS});

  circleShape.setFillColor(*TARGETED_COUNT ? sf::Color::Yellow : sf::Color::Green);

  circleShape.setOutlineThickness(2);
  circleShape.setOutlineColor(sf::Color::Black);

  circleShape.setPosition(m_position(0), m_position(1));
  target.draw(circleShape, states);

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

  text.setString(std::to_string(m_berryCount));

  auto textRect = text.getLocalBounds();
  text.setOrigin(textRect.left + textRect.width/2.0f,
                 textRect.top  + textRect.height/2.0f);
  text.setPosition(m_position(0), m_position(1));

  target.draw(text, states);
}
