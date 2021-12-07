#pragma once

#include "Grid.hpp"
#include "StaticBody.hpp"

#include <type_traits>
#include <random>

#include <tuple>
#include <set>

namespace details
{
  bool inside(Eigen::Vector2d position, Eigen::Vector2d topLeft, Eigen::Vector2d bottomRight)
  {
    return position(0) >= topLeft(0) && position(0) <= bottomRight(0) &&
           position(1) >= topLeft(1) && position(1) <= bottomRight(1);
  }
}

template<typename T, typename PRNG, typename InsertionFunc>
void generateClusters(const Grid<T>& grid, PRNG& prng, double radius, size_t clusterCount, size_t clusterSizeMin, size_t clusterSizeMax, InsertionFunc insertionFunc)
{
  static_assert(std::is_base_of_v<StaticBody, T>);

  auto gridPosition = grid.position();
  auto gridDimension = grid.dimension();

  std::uniform_real_distribution<double> xPositionDistribution(gridPosition(0), gridPosition(0) + gridDimension(0));
  std::uniform_real_distribution<double> yPositionDistribution(gridPosition(1), gridPosition(1) + gridDimension(1));
  std::uniform_real_distribution<double> distanceDistribution(2.0 * radius, 4.0 * radius);
  std::uniform_real_distribution<double> angleDistribution(-M_PI, M_PI);
  std::uniform_int_distribution<size_t> clusterSizeDistribution(clusterSizeMin, clusterSizeMax);

  for(size_t i=0; i<clusterCount; ++i)
  {
    size_t clusterSize = clusterSizeDistribution(prng);

    auto comparator = [](const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs) {
      return std::tie(lhs(0), lhs(1)) < std::tie(rhs(0), rhs(1));
    };
    std::set<Eigen::Vector2d, decltype(comparator)> positions(comparator); // Using a set for now even though it may not be the most efficient

    auto center = Eigen::Vector2d(xPositionDistribution(prng), yPositionDistribution(prng));
    insertionFunc(StaticBody(center, radius));
    positions.insert(center);

    static constexpr size_t MAX_TRIAL = 50;
    size_t failCount = 0;

    while(positions.size() != clusterSize && failCount != MAX_TRIAL)
    {
      std::uniform_int_distribution<size_t> indexDistribution(0, positions.size()-1);
      Eigen::Vector2d position = *std::next(positions.begin(), indexDistribution(prng));

      // Following is poisson disk sampling
      auto distance = distanceDistribution(prng);
      auto angle = angleDistribution(prng);

      Eigen::Vector2d newPosition = position + distance * Eigen::Vector2d(std::cos(angle), std::sin(angle));
      if(!details::inside(newPosition, gridPosition, gridPosition + gridDimension))
      {
        ++failCount;
        continue;
      }

      // TODO: Implement short-circuiting logic
      bool failed = false;
      grid.query(newPosition, 2.0 * radius, [&](const StaticBody& staticBody){
        if((staticBody.position() - newPosition).squaredNorm() < 4.0 * radius * radius)
          failed = true;
      });
      if(failed)
      {
        ++failCount;
        continue;
      }

      insertionFunc(StaticBody(newPosition, radius));
      positions.insert(newPosition);
    }
  }
}

template<typename T, typename PRNG, typename InsertionFunc>
[[gnu::noinline]] void generateNormal(PRNG& prng, double radius, size_t count, InsertionFunc insertionFunc)
{
  auto gridPosition = grid.position();
  auto gridDimension = grid.dimension();

  std::uniform_real_distribution<double> xPositionDistribution(gridPosition(0), gridPosition(0) + gridDimension(0));
  std::uniform_real_distribution<double> yPositionDistribution(gridPosition(1), gridPosition(1) + gridDimension(1));

  for(size_t i=0; i<count; ++i)
  {
    auto position = Eigen::Vector2d(xPositionDistribution(prng), yPositionDistribution(prng));
    insertionFunc(StaticBody(position, radius));
  }
}
