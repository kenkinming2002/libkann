#pragma once

#include "Agent.hpp"

struct MatchResult
{
  double score1;
  double score2;
};

[[nodiscard]] MatchResult matchSingle(Agent& lhs, Agent& rhs, size_t turnLimit);
[[nodiscard]] MatchResult match(Agent& lhs, Agent& rhs, size_t turnLimit);
