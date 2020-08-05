#include "Population.hpp"

#include <iostream>
#include <chrono>

int main()
{
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  Population population(seed, 100);
  population.select();
}
