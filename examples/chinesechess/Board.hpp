#pragma once

#include <stdint.h>
#include <stddef.h>

#include <ostream>
#include <array>
#include <vector>
#include <optional>
#include <tuple>

class Board
{
public:
  static constexpr size_t WIDTH = 9;
  static constexpr size_t HEIGHT = 10;

public:
  /* 
   * Prabably can pack to 2 uint8_t into 1 to save to memory usage and improve
   * memory locality but that will have to do for now.
   */
  struct Position 
  { 
  public:
    uint8_t x, y; 

  public:
    bool operator==(const Position& other) const = default;
  };

  struct Offset { int8_t x, y; };
  friend std::optional<Position> operator+(const Position& position, const Offset& offset)
  {
    auto x = static_cast<int>(position.x) + static_cast<int>(offset.x);
    auto y = static_cast<int>(position.y) + static_cast<int>(offset.y);
    if(x>=0 && x<WIDTH && y>=0 && y<HEIGHT)
      return Position{static_cast<uint8_t>(x), static_cast<uint8_t>(y)};
    else
      return std::nullopt;
  }

  struct Move 
  { 
  public:
    Position src, dst; 

  public:
    bool operator==(const Move& other) const = default;
  };

  struct Cell
  {
  public:
    enum class Type : uint8_t { EMPTY, CHARIOT, KNIGHT, ELEPHANT, /* weird name, I know */ ADVISOR, GENERAL, CANNON, SOLDIER } type;
    enum class Color : uint8_t { RED, BLACK, NONE } color;

  public:
    bool operator==(const Cell& other) const = default;

  public:
    constexpr Cell(Type type = Type::EMPTY, Color color = Color::NONE) : type(type), color(color) {}
  };

  /*
   * This class will not look ahead to see if you would loss.
   * Only after your general were taken will you be declared loss.
   */
  enum class State
  {
    WON,
    LOST,
    UNKNOWN
  };

public:
  using Row = std::array<Cell, WIDTH>;
  using Cells = std::array<Row, HEIGHT>;

public:
  Board();

public:
  Cell& cell(Position position) { return m_cells[position.y][position.x]; }
  const Cell& cell(Position position) const { return m_cells[position.y][position.x]; }

public:
  std::vector<Move> enumerateMove(Cell::Color color) const;
  bool validateMove(Move move) const;
  State performMove(Move move);

public:
  std::optional<Move> getMove(Position srcPosition, Offset dstOffset, Offset obstacleOffset) const;
  std::optional<Move> getMove(Position srcPosition, Offset dstOffset) const;
  static bool inSquare(Position position, Cell::Color color);
  static bool acrossRiver(Position position, Cell::Color color);

public:
  std::pair<double, double> estimateScore() const;

public:
  // Temporary only
  friend std::ostream& operator<<(std::ostream& os, const Board& board);

private:
  Cells m_cells;
};

