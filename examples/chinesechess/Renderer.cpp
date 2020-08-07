#include "Renderer.hpp"

#include <SFML/Graphics/Sprite.hpp>

namespace
{
  sf::Sprite makeSprite(sf::Texture& texture, sf::FloatRect dstRect)
  {
    auto textureSize = texture.getSize();
    sf::Sprite sprite(texture);
    sprite.setPosition(dstRect.left, dstRect.top);
    sprite.setScale(dstRect.width/textureSize.x, dstRect.height/textureSize.y);
    return sprite;
  }

  sf::Sprite makeSprite(sf::Texture& texture, sf::IntRect srcRect, sf::FloatRect dstRect)
  {
    sf::Sprite sprite(texture, srcRect);
    sprite.setPosition(dstRect.left, dstRect.top);
    sprite.setScale(dstRect.width / srcRect.width, dstRect.height/srcRect.height);
    return sprite;
  }
}


Renderer::Renderer(sf::RenderTarget& renderTarget)
  : m_renderTarget(renderTarget) 
{
  if(!m_piecesTexture.loadFromFile("resources/sprites/xiangqi-pieces-sprites.png"))
    throw std::runtime_error("Failed to load xingqi sprites");

  if(!m_boardTexture.loadFromFile("resources/sprites/board.jpg"))
    throw std::runtime_error("Failed to load xingqi sprites");
}

void Renderer::draw(const Board& board)
{
  const auto textureSize = m_piecesTexture.getSize();
  const auto spriteSize = sf::Vector2f(textureSize.x/7, textureSize.y/2);

  m_renderTarget.clear(sf::Color::White);

  auto sprite = makeSprite(m_boardTexture, sf::FloatRect(m_boardPosition, m_boardSize));
  m_renderTarget.draw(sprite);

  // Draw pieces
  for(uint8_t y=0; y<Board::HEIGHT; ++y)
    for(uint8_t x=0; x<Board::WIDTH; ++x)
    {
      const auto cell = board.cell(Board::Position{x, y});

      size_t spriteX, spriteY;
      switch(cell.type)
      {
      case Board::Cell::Type::GENERAL:
        spriteX = 0;
        break;
      case Board::Cell::Type::ADVISOR:
        spriteX = 1;
        break;
      case Board::Cell::Type::KNIGHT:
        spriteX = 2;
        break;
      case Board::Cell::Type::ELEPHANT:
        spriteX = 3;
        break;
      case Board::Cell::Type::CHARIOT:
        spriteX = 4;
        break;
      case Board::Cell::Type::CANNON:
        spriteX = 5;
        break;
      case Board::Cell::Type::SOLDIER:
        spriteX = 6;
        break;
      default:
        continue;
      }
      switch(cell.color)
      {
      case Board::Cell::Color::RED:
        spriteY = 0;
        break;
      case Board::Cell::Color::BLACK:
        spriteY = 1;
        break;
      default:
        continue;
      }

      auto srcRect = sf::IntRect(spriteX * spriteSize.x, spriteY * spriteSize.y, spriteSize.x, spriteSize.y);
      auto position = m_piecesPosition + m_piecesSize * sf::Vector2f(x, y);
      auto dstRect = sf::FloatRect(position, sf::Vector2f(m_piecesSize, m_piecesSize));
      auto sprite = makeSprite(m_piecesTexture, srcRect, dstRect);

      m_renderTarget.draw(sprite);
    }
}

std::optional<Board::Position> Renderer::screenToBoardCoords(sf::Vector2i screenCoords) const
{
  auto worldCoords = m_renderTarget.mapPixelToCoords(screenCoords);
  auto boardCoords = sf::Vector2i((worldCoords - m_piecesPosition) / m_piecesSize);
  if(boardCoords.x<0 || boardCoords.x>=Board::WIDTH)
    return std::nullopt;
  if(boardCoords.y<0 || boardCoords.y>=Board::HEIGHT)
    return std::nullopt;
  return Board::Position{static_cast<uint8_t>(boardCoords.x), static_cast<uint8_t>(boardCoords.y)};
}

