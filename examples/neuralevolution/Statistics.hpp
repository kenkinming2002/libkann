#pragma ocne

#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

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
  T minimum, median, maximum;
};

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

  std::sort(m_buffer.begin(), m_buffer.end());
  this->minimum = m_buffer.front();

  this->median = (m_buffer.size() % 2) == 0 
    ? (m_buffer[m_buffer.size()/2-1] + m_buffer[m_buffer.size()/2]) / 2
    : m_buffer[(m_buffer.size()-1)/2];

  this->maximum = m_buffer.back();

  m_buffer.clear();
}

template<typename Result, typename T>
std::string Statistics<Result, T>::toString() const
{
  std::stringstream ss;
  ss << std::fixed << std::setprecision(4);
  ss << mean << "/" << standardDeviation << "/";
  ss << minimum << "/" << median << "/" << maximum;
  ss << "(Mean/StandardDeviation/Minimum/Median/Maximum)";
  return ss.str();
}

template<typename Result, typename T>
std::vector<T> Statistics<Result, T>::m_buffer;
