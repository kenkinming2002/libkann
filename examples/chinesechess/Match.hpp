#pragma once

#include "Agent.hpp"
#include "Human.hpp"
#include "Board.hpp"

enum class PlayerID { NONE, AGENT1, AGENT2};

struct MatchSingleResult { PlayerID winningAgent; Board board; };
struct MatchResult { PlayerID winningAgent; Board board1; Board board2; };

MatchSingleResult matchSingle(Agent& lhs, Agent& rhs, size_t turnLimit = 100000);
MatchResult match(Agent& lhs, Agent& rhs, size_t turnLimit = 100000);
