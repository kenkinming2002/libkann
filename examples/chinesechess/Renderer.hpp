#pragma once

#include "Board.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/System/Vector2.hpp>

#include <optional>

class Renderer
{
public:
  Renderer(sf::RenderTarget& renderTarget);

public:
  void draw(const Board& board);
  std::optional<Board::Position> screenToBoardCoords(sf::Vector2i screenCoords) const;

public:
  const sf::Vector2f m_boardPosition = sf::Vector2f(50.0f, 50.0f);
  const sf::Vector2f m_boardSize     = sf::Vector2f(800.0f, 900.0f);

  const sf::Vector2f m_piecesPosition = sf::Vector2f(0.0f, 0.0f);
  const float        m_piecesSize     = 100.0f;

private:
  sf::RenderTarget& m_renderTarget;

private:
  sf::Texture m_piecesTexture;
  sf::Texture m_boardTexture;

};
