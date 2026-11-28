"""Student agent: edit ONLY this file."""
import sys
import math

from matplotlib.pyplot import margins

from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, check_move_validity, LINES)
from constants import *
import numpy as np
import time

class SearchTimeout(Exception):
    pass

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
    moves = list(get_valid_moves(board, reserves, player))

    if not moves:
        return None

    best = moves[0]
    depth = 1
    deadline = start + STEP_TIME_LIMIT - margin
    # Debug
    # print((time.perf_counter() - start) < STEP_TIME_LIMIT - margin)
    # Debug

    while time.perf_counter() < deadline:
        moves.sort(key=lambda m: m != best)
        best_value = float(-math.inf)
        cur_best = best

        alpha = float(-math.inf)
        beta = float(math.inf)

        try:
            for m in moves:
                if time.perf_counter() >= deadline:
                    raise SearchTimeout()

                b, r = copy_state(board, reserves)
                execute_move(b, r, player, m)

                value = minimax(b, r, player, opponent, alpha, beta, False, depth - 1, deadline, 1)

                if value > best_value:
                    best_value = value
                    cur_best = m

                alpha = max(alpha, best_value)
        except SearchTimeout:
            break

        best = cur_best
        depth += 1

    return best

WIN_SCORE = 100000

def minimax(board, reserves, player, opponent, alpha, beta, player_turn, depth, deadline, ply):
    if time.perf_counter() >= deadline:
        raise SearchTimeout()

    last_mover = opponent if player_turn else player
    winner= check_winner(board, last_mover)

    if winner == player:
        return WIN_SCORE - ply
    elif winner == opponent:
        return -WIN_SCORE + ply

    if depth == 0:
        return evaluate_move(board, reserves, player, opponent)

    current_player = player if player_turn else opponent
    moves = get_valid_moves(board, reserves, current_player)

    if not moves:
        if ALLOW_MOVES:
            winner = opponent if player_turn else player
            return WIN_SCORE - ply if winner == player else -WIN_SCORE + ply

        other_player = opponent if current_player == player else player

        if not get_valid_moves(board, reserves, other_player):
            return 0

        return minimax(board, reserves, player, opponent, alpha, beta, not player_turn, depth - 1, deadline, ply + 1)

    score = -math.inf if player_turn else math.inf
    for m in moves:
        if time.perf_counter() >= deadline:
            raise SearchTimeout()

        b, r = copy_state(board, reserves)
        execute_move(b, r, current_player, m)

        cur = minimax(b, r, player, opponent, alpha, beta, not player_turn, depth - 1, deadline, ply + 1)

        if player_turn:
            score = max(score, cur)
            alpha = max(alpha, score)

        else:
            score = min(score, cur)
            beta = min(beta, score)

        if beta <= alpha:
            break

    return score

LINE_WEIGHTS = [0, 1, 10, 100, 1000]

def evaluate_move(board, reserves, player, opponent):
    owners, sizes = top_view(board)
    score = 0

    for line in LINES[WIN_LENGTH]:
        cells = [owners[r, c] for r, c in line]

        own = sum(cell == player for cell in cells)
        enemy = sum(cell == opponent for cell in cells)

        # Obstacle makes the line unwinnable
        if OBSTACLE in cells:
            continue

        # Both players occupy the line
        if own > 0 and enemy > 0:
            continue

        if own > 0:
            score += LINE_WEIGHTS[own]

        elif enemy > 0:
            score -= LINE_WEIGHTS[enemy]

    return score