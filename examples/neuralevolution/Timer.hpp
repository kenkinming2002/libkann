#pragma once

#include <algorithm>
#include <numeric>

#include <SFML/System/Clock.hpp>

template<size_t SAMPLE_COUNT = 100>
class Timer
{
public:
  Timer() : m_samples{} {}

public:
  void begin()
  {
    m_clock.restart();
  }

  void end()
  {
    std::rotate(m_samples.begin(), m_samples.begin()+1, m_samples.end());
    m_samples.back() = m_clock.getElapsedTime().asSeconds();
  }

public:
  float average() const
  {
    return std::accumulate(m_samples.begin(), m_samples.end(), 0.0f) / m_samples.size();
  }

private:
  std::array<float, SAMPLE_COUNT> m_samples;
  sf::Clock m_clock;
};
