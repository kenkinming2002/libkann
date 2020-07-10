#pragma once

#include <Eigen/Eigen>

#include <vector>
#include <type_traits>
#include <list>

#include <iostream>

struct Box
{
public:
  Box(Eigen::Vector2d position, Eigen::Vector2d dimension) : m_position(position), m_dimension(dimension) {}

public:
  auto position() const { return m_position; }
  auto dimension() const { return m_dimension; }

private:
  Eigen::Vector2d m_position, m_dimension;
};

struct DividedBox : public Box
{
public:
  DividedBox(Eigen::Vector2d position, Eigen::Vector2d dimension, double divisionLength) 
    : Box(position, dimension), m_divisionLength(divisionLength),
      m_width(static_cast<size_t>(dimension(0) / m_divisionLength)+1),
      m_height(static_cast<size_t>(dimension(1) / m_divisionLength)+1) {}

public:
  size_t width() const { return m_width; }
  size_t height() const { return m_height; }

public:
  std::pair<size_t, size_t> indices(Eigen::Vector2d position) const
  {
    Eigen::Vector2d localPosition = (position - this->position()) / m_divisionLength;
    return std::make_pair(localPosition(0), localPosition(1));
  }

public:
  std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>> indices(Box box) const
  {
    auto clamp = [this](Eigen::Vector2d v){
      return Eigen::Vector2d(
        std::clamp(v(0), position()(0), position()(0) + dimension()(0)),
        std::clamp(v(1), position()(1), position()(1) + dimension()(1))
      );
    };

    auto topLeft = indices(clamp(box.position()));
    auto bottomRight = indices(clamp(box.position() + box.dimension()));

    return std::make_pair(topLeft, bottomRight);
  }

private:
  double m_divisionLength;
  size_t m_width, m_height;
};

template<typename T>
class Grid
{
public:
  using cell_type = std::list<T>;
  using iterator = typename cell_type::iterator;
  using const_iterator = typename cell_type::const_iterator;

public:
  static constexpr struct CenteredTag{} centerd_tag = {};

public:
  Grid(Eigen::Vector2d position, Eigen::Vector2d dimension, double divisionLength)
    : m_dividedBox(position, dimension, divisionLength), m_cells(m_dividedBox.width() * m_dividedBox.height()) {}

  Grid(CenteredTag /*tag*/, Eigen::Vector2d position, Eigen::Vector2d dimension, double divisionLength)
    : m_dividedBox(position - dimension/2.0, dimension, divisionLength), m_cells(m_dividedBox.width() * m_dividedBox.height()) {}

public:
  cell_type& cell(size_t x, size_t y)
  {
    size_t index = y*m_dividedBox.width()+x;
    assert(index < m_cells.size());
    return m_cells[index];
  }

  const cell_type& cell(size_t x, size_t y) const
  {
    size_t index = y*m_dividedBox.width()+x;
    assert(index < m_cells.size());
    return m_cells[index];
  }

  cell_type& cell(Eigen::Vector2d position)
  {
    auto [x, y] = m_dividedBox.indices(position);
    return cell(x, y);
  }

  const cell_type& cell(Eigen::Vector2d position) const
  {
    auto [x, y] = m_dividedBox.indices(position);
    return cell(x, y);
  }

public:
  /*
   * This class is *stupid* and do not know about the actual position of the
   * element or how to access it. Instead, you'll have to pass along the
   * position yourself. This follow the KISS principle and is much more
   * maintainable then other solution such as poassing along a accessor function
   * or having the user element inherit from a common base class.
   */
  void insert(Eigen::Vector2d position, const T& t) { cell(position).push_back(t); }
  void insert(Eigen::Vector2d position, T&& t) { cell(position).push_back(std::move(t)); }

  template<typename PositionFunc, typename InputIterator>
  void insert(PositionFunc positionFunc, InputIterator first, InputIterator last) 
  { 
    std::for_each(first, last, [this, &positionFunc](auto&& t){
      Eigen::Vector2d position = positionFunc(t);
      this->insert(position, std::forward<decltype(t)>(t)); 
    });
  }

  template<typename... Args>
  void emplace(Eigen::Vector2d position, Args&&... args) { cell(position).emplace_back(std::forward<Args>(args)...); }

  void erase(Eigen::Vector2d position, const_iterator it) { cell(position).erase(it); }

public:
  /*
   * The following function all return a vector of reference of some sort, which
   * incurs memory allocation at each call. 
   *
   * This may be circumvented by using the same vector and clearing it to reuse
   * the allocated memory instead of frequently calling new and delete, but this
   * add to code complexity and may have concurency issues.
   *
   * Another solution is to use a custom iterator, which would avoid all memory
   * allocation but this leads to a lot more boilerplate code to maintain. With
   * that said, if there is performance issues, this will be the solution or c++
   * coroutine will be.
   */
  auto all()
  {
    std::vector<std::reference_wrapper<T>> result;
    for(auto& cell: m_cells)
      for(auto& t: cell)
        result.push_back(std::ref(t));

    return result;
  }

  auto all() const
  {
    std::vector<std::reference_wrapper<const T>> result;
    for(auto& cell: m_cells)
      for(auto& t: cell)
        result.push_back(std::cref(t));

    return result;
  }

  auto query(Box box)
  {
    auto [topLeft, bottomRight] = m_dividedBox.indices(box);

    auto [topLeftX, topLeftY] = topLeft;
    auto [bottomRightX, bottomRightY] = bottomRight;

    std::vector<std::reference_wrapper<T>> result;
    for(size_t y=topLeftY; y<=bottomRightY; ++y)
      for(size_t x=topLeftX; x<=bottomRightX; ++x)
        for(auto& t: cell(x, y))
          result.push_back(std::ref(t));

    return result;
  }

  auto query(Box box) const
  {
    auto [topLeft, bottomRight] = m_dividedBox.indices(box);

    auto [topLeftX, topLeftY] = topLeft;
    auto [bottomRightX, bottomRightY] = bottomRight;

    std::vector<std::reference_wrapper<const T>> result;
    for(size_t y=topLeftY; y<=bottomRightY; ++y)
      for(size_t x=topLeftX; x<=bottomRightX; ++x)
        for(const auto& t: cell(x, y))
          result.push_back(std::cref(t));

    return result;
  }

public:
  template<typename Predicate>
  void remove_if(Predicate predicate)
  {
    std::vector<std::reference_wrapper<const T>> result;
    for(auto& cell: m_cells)
      cell.remove_if(predicate);
  }

public:
  /*
   * Retrieve position of all element and relocate them accordingly.
   * Must be called after position of any element are modified before any more
   * query or result may be inaccurate.
   *
   * In the future, a per-element version may be implemented
   */
  template<typename PositionFunc>
  void synchronize(PositionFunc positionFunc)
  {
    cell_type orphans;
    for(auto& cell: m_cells)
    {
      auto it = cell.begin();
      while(it != cell.end())
      {
        auto& newCell = this->cell(positionFunc(*it));
        if(&cell != &newCell)
        {
          auto itCopy = it++;
          orphans.splice(orphans.end(), cell, itCopy);
        }
        else
          ++it;
      }
    }

    auto it = orphans.begin();
    while(it != orphans.end())
    {
      auto& newCell = this->cell(positionFunc(*it));
      auto itCopy = it++;
      newCell.splice(newCell.end(), orphans, itCopy);
    }
  }

public:
  DividedBox m_dividedBox;
  std::vector<cell_type> m_cells;
};

template class Grid<int>;
