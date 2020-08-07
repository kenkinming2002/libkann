#pragma once

#include "Agent.hpp"
#include "Board.hpp"

enum class AgentID { NONE, _1, _2 };

struct MatchSingleResult { AgentID winningAgent; Board board; };
MatchSingleResult matchSingle(Agent& lhs, Agent& rhs, size_t turnLimit = 100000);

struct MatchResult { AgentID winningAgent; Board board1; Board board2; };
MatchResult match(Agent& lhs, Agent& rhs, size_t turnLimit = 100000);

