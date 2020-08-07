#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>
#include <tuple>

template<typename Result, typename T>
struct Statistics
{
private:
  static std::vector<T> m_buffer;

public:
  Statistics() : mean(), standardDeviation(), minimum(), median(), maximum() {}

  // Remark: not MT-safe
  template<typename InputIterator, typename UnaryOperation>
  Statistics(InputIterator first, InputIterator last, UnaryOperation op);

public:
  std::string toString() const;

public:
  Result mean;
  Result standardDeviation; 
  T minimum, lowerQuartile, median, upperQuartile, maximum;
};

namespace details
{
  template<typename Iterator>
  using Range = std::pair<Iterator, Iterator>;

  template<typename RandomAccessIterator>
  auto median(RandomAccessIterator first, RandomAccessIterator last) 
    -> std::tuple<Range<RandomAccessIterator>, Range<RandomAccessIterator>, typename std::iterator_traits<RandomAccessIterator>::value_type> 
  {
    size_t n = std::distance(first, last);
    if(n % 2 == 0)
    {
      auto middleFirst = std::next(first, n/2-1);
      auto middleLast  = std::next(first, n/2);
      return {{first, middleLast}, {middleLast, last}, (*middleFirst + *middleLast) / 2};
    }
    else
    {
      auto middle = std::next(first, (n-1)/2);
      return {{first, std::next(middle)}, {middle, last}, *middle};
    }
  }
}


template<typename Result, typename T>
template<typename InputIterator, typename UnaryOperation>
Statistics<Result, T>::Statistics(InputIterator first, InputIterator last, UnaryOperation op)
{
  std::transform(first, last, std::back_inserter(m_buffer), op);

  T total = std::accumulate(m_buffer.begin(), m_buffer.end(), T());
  this->mean = static_cast<Result>(total) / m_buffer.size();

  T totalVariance = std::accumulate(m_buffer.begin(), m_buffer.end(), T(), [](const T& accumulator, const T& val){
      return accumulator + val * val;
  });
  this->standardDeviation = std::sqrt(static_cast<Result>(totalVariance) / m_buffer.size());

  if(!m_buffer.empty())
  {
    std::sort(m_buffer.begin(), m_buffer.end());
    this->minimum = m_buffer.front();

    auto [lowerRange, upperRange, median] = details::median(m_buffer.begin(), m_buffer.end());
    this->median = median;
    std::tie(std::ignore, std::ignore,this->lowerQuartile) = details::median(lowerRange.first, lowerRange.second);
    std::tie(std::ignore, std::ignore,this->upperQuartile) = details::median(upperRange.first, upperRange.second);

    this->maximum = m_buffer.back();
  }

  m_buffer.clear();
}

template<typename Result, typename T>
std::string Statistics<Result, T>::toString() const
{
  std::stringstream ss;
  ss << std::fixed << std::setprecision(4);

  ss << mean << "/" << standardDeviation << "(Mean/StandardDeviation);";
  ss << minimum << "/" << lowerQuartile << "/" << median << "/" << upperQuartile << "/" << maximum << "(Minimum/LQ/Median/UQ/Maximum)";
  return ss.str();
}

template<typename Result, typename T>
std::vector<T> Statistics<Result, T>::m_buffer;
