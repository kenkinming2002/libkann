#pragma once

#include <Eigen/Eigen>

#include <vector>
#include <type_traits>

#include <iostream>

namespace
{
  auto clamp(Eigen::Vector2d v, Eigen::Vector2d lo, Eigen::Vector2d hi) 
  {
    return Eigen::Vector2d(
      std::clamp(v(0), lo(0), hi(0)),
      std::clamp(v(1), lo(1), hi(1))
    );
  };
}

struct Box
{
public:
  Box(Eigen::Vector2d position, Eigen::Vector2d dimension) : m_position(position), m_dimension(dimension) {}

public:
  auto position() const { return m_position; }
  auto dimension() const { return m_dimension; }

  double distance(Eigen::Vector2d point) const 
  {
    return (point - clamp(point, m_position, m_position + m_dimension)).norm();
  }

  double squaredDistance(Eigen::Vector2d point) const 
  {
    return (point - clamp(point, m_position, m_position + m_dimension)).squaredNorm();
  }


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
    auto topLeft     = indices(clamp(box.position()                  , this->position(), this->position() + this->dimension()));
    auto bottomRight = indices(clamp(box.position() + box.dimension(), this->position(), this->position() + this->dimension()));

    return std::make_pair(topLeft, bottomRight);
  }

  Box box(size_t x, size_t y) const
  {
    return Box(Eigen::Vector2d(x * m_divisionLength, y * m_divisionLength) + this->position(), 
        Eigen::Vector2d(m_divisionLength, m_divisionLength));
  }

private:
  double m_divisionLength;
  size_t m_width, m_height;
};

template<typename T>
class Grid
{
public:
  using cell_type = std::vector<T>;
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

  /*
   * The following 4 functions are const and non const overload of 2 types of
   * query function, one which return all entities that possibly lie in a box
   * while the other a circle. Notice the word possibly as it it not a guarantee
   * and you'll have to check for yourself.
   *
   * Currently, only the circle version are used as it replaced the usage of the
   * box version and return less spurious result and will reduce the number of
   * further testing we have to do(raycast in particular) as a circle is always
   * contained in a box. However, there is one major flaw and that is the
   * presence of a branch in the circle version, and its unpreditable nature. As
   * branch misprediction is costly, the performance benefit of less testing to
   * is outweighted and the performace gain is negligible. Anyway, the circle
   * version is easy enough to implement that it does not matter. However, it
   * may be an idea to switch to the branchless box version. Only time will
   * tell.
   */
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

  template<typename Callback>
  void query(Eigen::Vector2d position, double radius, Callback callback) const
  {
    auto halfDimension = Eigen::Vector2d(radius, radius);
    auto box = Box(position - halfDimension, 2.0 * halfDimension);

    auto [topLeft, bottomRight] = m_dividedBox.indices(box);

    auto [topLeftX, topLeftY] = topLeft;
    auto [bottomRightX, bottomRightY] = bottomRight;

    for(size_t y=topLeftY; y<=bottomRightY; ++y)
      for(size_t x=topLeftX; x<=bottomRightX; ++x)
      {
        auto subBox = m_dividedBox.box(x, y);
        if(subBox.squaredDistance(position) <= radius * radius)
          for(const auto& t: cell(x, y))
            callback(t);
      }
  }

  template<typename Callback>
  void query(Eigen::Vector2d position, double radius, Callback callback)
  {
    auto halfDimension = Eigen::Vector2d(radius, radius);
    auto box = Box(position - halfDimension, 2.0 * halfDimension);

    auto [topLeft, bottomRight] = m_dividedBox.indices(box);

    auto [topLeftX, topLeftY] = topLeft;
    auto [bottomRightX, bottomRightY] = bottomRight;

    for(size_t y=topLeftY; y<=bottomRightY; ++y)
      for(size_t x=topLeftX; x<=bottomRightX; ++x)
      {
        auto subBox = m_dividedBox.box(x, y);
        if(subBox.squaredDistance(position) <= radius * radius)
          for(auto& t: cell(x, y))
            callback(t);
      }
  }

public:
  template<typename Predicate>
  size_t remove_if(Predicate predicate)
  {
    size_t count = 0;
    std::vector<std::reference_wrapper<const T>> result;
    for(auto& cell: m_cells)
    {
      size_t oldSize = cell.size();
      cell.erase(std::remove_if(cell.begin(), cell.end(), predicate), cell.end());
      size_t newSize = cell.size();

      count += oldSize - newSize;
    }
    return count;
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
      auto it = std::partition(cell.begin(), cell.end(), [&](const auto& t){
          auto& newCell = this->cell(positionFunc(t));
          return &cell == &newCell;
      });
      std::move(it, cell.end(), std::back_inserter(orphans));
      cell.erase(it, cell.end());
    }

    for(auto& t: orphans)
    {
      auto& newCell = this->cell(positionFunc(t));
      newCell.push_back(std::move(t));
    }
    orphans.clear();
  }

public:
  DividedBox m_dividedBox;
  std::vector<cell_type> m_cells;
};
