#include "Human.hpp"

#include <stdexcept>
#include <iostream>

#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Event.hpp>

static constexpr unsigned WINDOW_WIDTH = 1000;
static constexpr unsigned WINDOW_HEIGHT = 1000;


Human::Human()
  : m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "App"),
    m_boardPosition(50.0f, 50.0f),
    m_boardSize(800.0f, 900.0f),
    m_piecesPosition(0.0f, 0.0f),
    m_piecesSize(100.0f)
{
  if(!m_piecesTexture.loadFromFile("resources/sprites/xiangqi-pieces-sprites.png"))
    throw std::runtime_error("Failed to load xingqi sprites");

  if(!m_boardTexture.loadFromFile("resources/sprites/board.jpg"))
    throw std::runtime_error("Failed to load xingqi sprites");
}



std::optional<Board::Position> Human::screenToBoardCoords(sf::Vector2i screenCoords) const
{
  auto worldCoords = m_window.mapPixelToCoords(screenCoords);

  auto boardCoords = sf::Vector2i((worldCoords - m_piecesPosition) / m_piecesSize);
  if(boardCoords.x<0 || boardCoords.x>=Board::WIDTH)
    return std::nullopt;
  if(boardCoords.y<0 || boardCoords.y>=Board::HEIGHT)
    return std::nullopt;

  return Board::Position{static_cast<uint8_t>(boardCoords.x), static_cast<uint8_t>(boardCoords.y)};
}

void Human::handleInput()
{
  sf::Event event;
  while (m_window.pollEvent(event))
  {
    switch(event.type)
    {
    case sf::Event::Closed:
      m_window.close();
      break;
    case sf::Event::MouseButtonPressed:
      if(auto boardCoords = screenToBoardCoords(sf::Vector2i(event.mouseButton.x, event.mouseButton.y)); boardCoords)
      {
        if(!m_src)
          m_src = boardCoords;
        else
          m_move = Board::Move(*m_src, *boardCoords);
      }
      break;
    default:
      break;
    }
  }
}

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

void Human::render(const Board& board)
{
  const auto textureSize = m_piecesTexture.getSize();
  const auto spriteSize = sf::Vector2f(textureSize.x/7, textureSize.y/2);

  m_window.clear(sf::Color::White);

  auto sprite = makeSprite(m_boardTexture, sf::FloatRect(m_boardPosition, m_boardSize));
  m_window.draw(sprite);

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

      m_window.draw(sprite);
    }


  m_window.display();
}


std::optional<Board::Move> Human::selectMove(const Board& board, Board::Cell::Color color)
{
  if(board.enumerateMove(color).empty())
    return std::nullopt;

  do
  {
    render(board);
    handleInput();

    if(m_move)
    {
      if(board.validateMove(*m_move))
      {
        return m_move;
      }
      else
      {
        m_src = std::nullopt;
        m_move = std::nullopt;
      }
    }
  } while(m_window.isOpen());

  return std::nullopt;
}

