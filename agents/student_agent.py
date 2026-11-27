"""Student agent: edit ONLY this file."""
import sys
import math

from matplotlib.pyplot import margins

from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, check_move_validity)
from constants import *
import numpy as np
import time

@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"

        weights = {
            "w1": 1,
            "w2:": 2,
            "w3": 3
        }

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        start = time.perf_counter()
        margin = 0.2
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.

        return iterative_deepening(board, reserves, player, opponent, start, margin)


def iterative_deepening(board, reserves, player, opponent, start, margin):
    best = random_move(board, reserves, player, np.random.default_rng())
    step_amount = 1

    # Debug
    # print((time.perf_counter() - start) < STEP_TIME_LIMIT - margin)
    # Debug

    while time.perf_counter() - start < STEP_TIME_LIMIT - margin:
        best_value = float(-math.inf)
        cur_best = best

        alpha = float(-math.inf)
        beta = float(math.inf)

        for m in get_valid_moves(board, reserves, player):
            if time.perf_counter() - start >= STEP_TIME_LIMIT - margin:
                break

            b, r = copy_state(board, reserves)
            execute_move(b, r, player, m)

            value = minimax(b, r, player, opponent, alpha, beta, False, step_amount)

            if value > best_value:
                best_value = value
                cur_best = m

            alpha = max(alpha, best_value)

        else:
            best = cur_best
            step_amount += 1

    return best


def minimax(board, reserves, player, opponent, alpha, beta, player_turn, steps):
    if steps == 0 or check_endgame(board, reserves, player if player_turn else opponent, steps * 2, ALLOW_MOVES): # or if game state end (winner)
        return evaluate_move(board, reserves, player_turn, player, opponent)

    if player_turn:
        score = -math.inf
    else:
        score = math.inf

    for m in get_valid_moves(board, reserves, player):
        b, r = copy_state(board, reserves)
        execute_move(b, r, player, m)

        if player_turn:
            cur = minimax(b, r, player, opponent, alpha, beta, False, steps - 1)

            score = max(score, cur)
            alpha = max(alpha, cur)

        else:
            cur = minimax(b, r, player, opponent, alpha, beta, True, steps - 1)

            score = min(score, cur)
            beta = min(beta, cur)

        if beta <= alpha:
            break

    return score


def evaluate_move(board, reserves, if_player, player, opponent):
    if if_player:
        p = player
    else:
        p = opponent

    state = top_view(board)

    

    return 0