#pragma once

#include "Board.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>

#include <SFML/System/Vector2.hpp>
#include <optional>

class Human
{
public:
  Human();

private:
  std::optional<Board::Position> screenToBoardCoords(sf::Vector2i screenCoords) const;

public:
  std::optional<Board::Move> selectMove(const Board& board, Board::Cell::Color color);

private:
  void handleInput();
  void render(const Board& board);

private:
  sf::RenderWindow m_window;

private:
  const sf::Vector2f m_boardPosition;
  const sf::Vector2f m_boardSize;

  const sf::Vector2f m_piecesPosition;
  const float m_piecesSize;

private:
  sf::Texture m_piecesTexture;
  sf::Texture m_boardTexture;

private:
  std::optional<Board::Position> m_src;
  std::optional<Board::Move>     m_move;
};
