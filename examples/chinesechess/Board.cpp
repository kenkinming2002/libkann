#include "Board.hpp"

#include <stdexcept>
#include <cassert>
#include <iostream>

/* 
 * I know, I know. You people are screaming here that macros a evil, but would
 * you rather stare at a unreasonably long line with repetitive information
 * instead of some macro that just work?
 * 
 * Just make sure to undefine it and it will be fine
 */
#define EMPTY_CELL Board::Cell(Board::Cell::Type::EMPTY, Board::Cell::Color::NONE)
#define CELL(type) Board::Cell(Board::Cell::Type::type, color)

namespace
{
  constexpr Board::Row makeEndRow(Board::Cell::Color color)
  {
    return {CELL(CHARIOT), CELL(KNIGHT), CELL(ELEPHANT), CELL(ADVISOR), CELL(GENERAL), CELL(ADVISOR), CELL(ELEPHANT), CELL(KNIGHT), CELL(CHARIOT)};
  }

  constexpr Board::Row makeSoldierRow(Board::Cell::Color color)
  {
    return {CELL(SOLDIER), EMPTY_CELL, CELL(SOLDIER), EMPTY_CELL, CELL(SOLDIER), EMPTY_CELL, CELL(SOLDIER), EMPTY_CELL, CELL(SOLDIER)};
  }

  constexpr Board::Row makeCannonRow(Board::Cell::Color color)
  {
    return {EMPTY_CELL, CELL(CANNON), EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, CELL(CANNON), EMPTY_CELL};
  }

  constexpr Board::Row makeEmptyRow(Board::Cell::Color /*color*/)
  {
    return {EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL};
  }
}
#undef CELL
#undef EMPTY_CELL

Board::Board()
  : m_cells{
      makeEndRow(Cell::Color::RED), 
      makeEmptyRow(Cell::Color::RED), 
      makeCannonRow(Cell::Color::RED), 
      makeSoldierRow(Cell::Color::RED), 
      makeEmptyRow(Cell::Color::RED),

      makeEmptyRow(Cell::Color::BLACK), 
      makeSoldierRow(Cell::Color::BLACK), 
      makeCannonRow(Cell::Color::BLACK), 
      makeEmptyRow(Cell::Color::BLACK), 
      makeEndRow(Cell::Color::BLACK)
    } {}

std::vector<Board::Move> Board::enumerateMove(Cell::Color color) const
{
  assert(color != Cell::Color::NONE);
  std::vector<Board::Move> moves;
  for(uint8_t y=0; y<HEIGHT; ++y)
    for(uint8_t x=0; x<WIDTH; ++x)
    {
      const auto srcPosition = Position{x, y};
      const auto& srcCell = cell(srcPosition);
      if(srcCell.color == color)
      {
        switch(srcCell.type)
        {
          case Board::Cell::Type::EMPTY:
            continue;
          case Board::Cell::Type::CHARIOT:
          {
            for(int dstX = static_cast<int>(x)-1; dstX>=0; --dstX)
            {
              const auto dstPosition = Position{static_cast<uint8_t>(dstX), y};
              const auto& dstCell = cell(dstPosition);
              if(dstCell.color == color)
                break;
              moves.push_back(Move{srcPosition, dstPosition});
              if(dstCell.color != Cell::Color::NONE)
                break;
            }
            for(int dstX = static_cast<int>(x)+1; dstX<static_cast<int>(WIDTH); ++dstX)
            {
              const auto dstPosition = Position{static_cast<uint8_t>(dstX), y};
              const auto& dstCell = cell(dstPosition);
              if(dstCell.color == color)
                break;
              moves.push_back(Move{srcPosition, dstPosition});
              if(dstCell.color != Cell::Color::NONE)
                break;
            }

            for(int dstY = static_cast<int>(y)-1; dstY>=0; --dstY)
            {
              const auto dstPosition = Position{x, static_cast<uint8_t>(dstY)};
              const auto& dstCell = cell(dstPosition);
              if(dstCell.color == color)
                break;
              moves.push_back(Move{srcPosition, dstPosition});
              if(dstCell.color != Cell::Color::NONE)
                break;
            }

            for(int dstY = static_cast<int>(y)+1; dstY<static_cast<int>(HEIGHT); ++dstY)
            {
              const auto dstPosition = Position{x, static_cast<uint8_t>(dstY)};
              const auto& dstCell = cell(dstPosition);
              if(dstCell.color == color)
                break;
              moves.push_back(Move{srcPosition, dstPosition});
              if(dstCell.color != Cell::Color::NONE)
                break;
            }
            break;
          }
          case Board::Cell::Type::KNIGHT:
          {
            static constexpr std::pair<Offset, Offset> offsets[] = {
              std::make_pair(Offset{-1, -2}, Offset{0, -1}),
              std::make_pair(Offset{ 1, -2}, Offset{0, -1}),
              std::make_pair(Offset{-2, -1}, Offset{-1, 0}),
              std::make_pair(Offset{-2,  1}, Offset{-1, 0}),

              std::make_pair(Offset{-1,  2}, Offset{0, 1}),
              std::make_pair(Offset{ 1,  2}, Offset{0, 1}),
              std::make_pair(Offset{ 2, -1}, Offset{1, 0}),
              std::make_pair(Offset{ 2,  1}, Offset{1, 0}),
            };
            for(auto [dstOffset, obstacleOffset] : offsets)
              if(auto move = getMove(srcPosition, dstOffset, obstacleOffset))
                moves.push_back(*move);
            break;
          }
          case Board::Cell::Type::ELEPHANT:
          {
            static constexpr std::pair<Offset, Offset> offsets[] = {
              std::make_pair(Offset{-2, -2}, Offset{-1, -1}),
              std::make_pair(Offset{ 2, -2}, Offset{ 1, -1}),
              std::make_pair(Offset{-2,  2}, Offset{-1,  1}),
              std::make_pair(Offset{ 2,  2}, Offset{ 1,  1}),
            };
            for(const auto& [dstOffset, obstacleOffset] : offsets)
              if(auto move = getMove(srcPosition, dstOffset, obstacleOffset))
                if(!acrossRiver(move->dst, color))
                  moves.push_back(*move);
            break;
          }
          case Board::Cell::Type::ADVISOR:
          {
            static constexpr Offset offsets[] = {
              Offset{-1, -1},
              Offset{ 1, -1},
              Offset{-1,  1},
              Offset{ 1,  1},
            };
            for(const auto& dstOffset : offsets)
              if(auto move = getMove(srcPosition, dstOffset))
                if(inSquare(move->dst, color))
                  moves.push_back(*move);
            break;
          }
          case Board::Cell::Type::GENERAL:
          {
            static constexpr Offset offsets[] = {
              Offset{-1,  0},
              Offset{ 1,  0},
              Offset{ 0, -1},
              Offset{ 0,  1},
            };
            for(const auto& dstOffset : offsets)
              if(auto move = getMove(srcPosition, dstOffset))
                if(inSquare(move->dst, color))
                  moves.push_back(*move);

            switch(color)
            {
            case Cell::Color::RED:
              for(int dstY=static_cast<int>(y)+1; dstY<static_cast<int>(HEIGHT); ++dstY)
              {
                const auto dstPosition = Position{x, static_cast<uint8_t>(dstY)};
                const auto& dstCell = cell(dstPosition);
                if(!dstCell.empty())
                {
                  if(dstCell.type == Cell::Type::GENERAL)
                    moves.push_back(Move{srcPosition, dstPosition});

                  break;
                }
              }
              break;
            case Cell::Color::BLACK:
              for(int dstY=static_cast<int>(y)-1; dstY>=0; --dstY)
              {
                const auto dstPosition = Position{x, static_cast<uint8_t>(dstY)};
                const auto& dstCell = cell(dstPosition);
                if(!dstCell.empty())
                {
                  if(dstCell.type == Cell::Type::GENERAL)
                    moves.push_back(Move{srcPosition, dstPosition});

                  break;
                }
              }
              break;
            default:
              break;
            }
            break;
          }
          case Board::Cell::Type::CANNON:
          {
            bool passedFirstPiece;

            passedFirstPiece = false;
            for(int dstX = static_cast<int>(x)-1; dstX>=0; --dstX)
            {
              const auto dstPosition = Position{static_cast<uint8_t>(dstX), y};
              const auto& dstCell = cell(dstPosition);
              if(!passedFirstPiece)
              {
                if(dstCell.color == Cell::Color::NONE)
                  moves.push_back(Move{srcPosition, dstPosition});
                else
                  passedFirstPiece = true;
              }
              else
              {
                if(dstCell.color != Cell::Color::NONE)
                {
                  if(dstCell.color != color)
                    moves.push_back(Move{srcPosition, dstPosition});
                  break;
                }
                else
                  continue;
              }
            }

            passedFirstPiece = false;
            for(int dstX = static_cast<int>(x)+1; dstX<WIDTH; ++dstX)
            {
              const auto dstPosition = Position{static_cast<uint8_t>(dstX), y};
              const auto& dstCell = cell(dstPosition);
              if(!passedFirstPiece)
              {
                if(dstCell.color == Cell::Color::NONE)
                  moves.push_back(Move{srcPosition, dstPosition});
                else
                  passedFirstPiece = true;
              }
              else
              {
                if(dstCell.color != Cell::Color::NONE)
                {
                  if(dstCell.color != color)
                    moves.push_back(Move{srcPosition, dstPosition});
                  break;
                }
                else
                  continue;
              }
            }

            passedFirstPiece = false;
            for(int dstY = static_cast<int>(y)-1; dstY>=0; --dstY)
            {
              const auto dstPosition = Position{x, static_cast<uint8_t>(dstY)};
              const auto& dstCell = cell(dstPosition);
              if(!passedFirstPiece)
              {
                if(dstCell.color == Cell::Color::NONE)
                  moves.push_back(Move{srcPosition, dstPosition});
                else
                  passedFirstPiece = true;
              }
              else
              {
                if(dstCell.color != Cell::Color::NONE)
                {
                  if(dstCell.color != color)
                    moves.push_back(Move{srcPosition, dstPosition});
                  break;
                }
                else
                  continue;
              }
            }

            passedFirstPiece = false;
            for(int dstY = static_cast<int>(y)+1; dstY<HEIGHT; ++dstY)
            {
              const auto dstPosition = Position{x, static_cast<uint8_t>(dstY)};
              const auto& dstCell = cell(dstPosition);
              if(!passedFirstPiece)
              {
                if(dstCell.color == Cell::Color::NONE)
                  moves.push_back(Move{srcPosition, dstPosition});
                else
                  passedFirstPiece = true;
              }
              else
              {
                if(dstCell.color != Cell::Color::NONE)
                {
                  if(dstCell.color != color)
                    moves.push_back(Move{srcPosition, dstPosition});
                  break;
                }
                else
                  continue;
              }
            }
            break;
          }
          case Board::Cell::Type::SOLDIER:
          {
            auto forwardOffset = color == Cell::Color::RED ? Offset{0, 1} : Offset{0, -1};
            if(auto move = getMove(srcPosition, forwardOffset))
              moves.push_back(*move);

            if(acrossRiver(srcPosition, color))
            {
              std::cout << "Hey\n";
              static constexpr Offset offsets[] = {Offset{-1,  0}, Offset{1,  0}};
              for(const auto& dstOffset : offsets)
                if(auto move = getMove(srcPosition, dstOffset))
                  moves.push_back(*move);
              break;
            }
            break;
          }
          default: 
            throw std::runtime_error("I don't know WTH happened");
        }
      }
    }

  return moves;
}

bool Board::validateMove(Move move, Cell::Color color) const
{
  // Whatever. Makes it work first optimize later.
  auto moves = this->enumerateMove(color);
  return std::find(moves.begin(), moves.end(), move) != moves.end();
}

Board::Cell Board::performMove(Move move)
{
  auto& srcCell = cell(move.src);
  auto& dstCell = cell(move.dst);

  // Basic perliminary sanity check
  assert(move.src.x != move.dst.x || move.src.y != move.dst.y);
  assert(srcCell.color != Cell::Color::NONE);
  assert(srcCell.color != dstCell.color);

  auto result = dstCell;

  dstCell = srcCell;
  srcCell = Cell();

  return result;
}

void Board::undoMove(Move move, Cell cell)
{
  auto& srcCell = this->cell(move.src);
  auto& dstCell = this->cell(move.dst);

  srcCell = dstCell;
  dstCell = cell;
}

std::optional<Board::Move> Board::getMove(Position srcPosition, Offset dstOffset, Offset obstacleOffset) const
{
  auto dstPosition = srcPosition + dstOffset;
  auto obstaclePosition = srcPosition + obstacleOffset;
  if(!dstPosition || !obstaclePosition)
    return std::nullopt;
  if(cell(*obstaclePosition).color != Cell::Color::NONE || cell(srcPosition).color == cell(*dstPosition).color)
    return std::nullopt;
  return Move{srcPosition, *dstPosition};
}

std::optional<Board::Move> Board::getMove(Position srcPosition, Offset dstOffset) const
{
  auto dstPosition = srcPosition + dstOffset;
  if(!dstPosition)
    return std::nullopt;
  if(cell(srcPosition).color == cell(*dstPosition).color)
    return std::nullopt;
  return Move{srcPosition, *dstPosition};
}

bool Board::inSquare(Position position, Cell::Color color)
{
  if(position.x<3 || position.x>=6)
    return false;

  switch(color)
  {
  case Cell::Color::RED:
    return position.y<3;
    break;
  case Cell::Color::BLACK:
    return position.y>=7;
  default:
    throw std::runtime_error("I don't know WTH happened");
  }
}

bool Board::acrossRiver(Position position, Cell::Color color)
{
  switch(color)
  {
  case Cell::Color::RED:
    return position.y>=5;
    break;
  case Cell::Color::BLACK:
    return position.y<5;
  default:
    throw std::runtime_error("I don't know WTH happened");
  }
}

std::pair<double, double> Board::estimateScore() const
{
  double redScore = 0.0, blackScore = 0.0;
  for(const auto& row: m_cells)
    for(const auto& cell: row)
    {
      auto score = [&](){
        switch(cell.type)
        {
        case Cell::Type::EMPTY:    return 0.0;
        case Cell::Type::CHARIOT:  return 2.0;
        case Cell::Type::KNIGHT:   return 1.0;
        case Cell::Type::ELEPHANT: return 1.0;
        case Cell::Type::ADVISOR:  return 1.0;
        case Cell::Type::GENERAL:  return 0.0;
        case Cell::Type::CANNON:   return 1.0;
        case Cell::Type::SOLDIER:  return 0.5;
        default: throw std::runtime_error("I don't know WTH happened");
        }
      };
      switch(cell.color)
      {
      case Cell::Color::RED: 
        redScore += score();
        break;
      case Cell::Color::BLACK: 
        blackScore += score(); 
        break;
      default:
        break;
      }
    }
  
  return std::make_pair(redScore, blackScore);
}

std::optional<Board::Position> operator+(const Board::Position& position, const Board::Offset& offset)
{
  auto x = static_cast<int>(position.x) + static_cast<int>(offset.x);
  auto y = static_cast<int>(position.y) + static_cast<int>(offset.y);
  if(x>=0 && x<Board::WIDTH && y>=0 && y<Board::HEIGHT)
    return Board::Position{static_cast<uint8_t>(x), static_cast<uint8_t>(y)};
  else
    return std::nullopt;
}
