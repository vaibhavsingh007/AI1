# This algorithm implements MDP Value Iteration using Bellman Eqn.
# vsingh38@uic.edu 112817
from enum import Enum
from pprint import pprint
import math
import copy     # For performing deep-copy

class Action(Enum):
    __order__ = 'none up down left right'
    none = 0
    up = 1
    down = 2
    left = 3
    right = 4

# Class that encapsulates the Markov Decision Process setup.
# MDP input components are: <S,A,T,R>, that will be used by the
#..Value Iteration routine below, to calculate the utilities.
class MDP(object):
    env = []        # Captures the environment (reward grid)
    rows = None    # Stores the dimensions of the grid
    cols = None
    policy = []   # To store the policy leading to
    
    def __init__(self, grid):
        # initial_puzzle expected to be a 2-D matrix > x0
        self.env = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols) if grid[i][j] != None]
        # Initialize the discount factor (used for convergence) in Bellman Eqn.
        self.gamma = 0.9
        
    # Returns P(s'|s,a), for all s'.
    # Does not perform bounds checking.
    def transition(self, s, a):
        p_stochastic_move = 0.1
        p_move = 0.8
        
        if (a == Action.none):
            return [(0.0, s)]
        
        if (a == Action.up) or (a == Action.down):
            return [(p_move, self.__move(s, a)),
                    (p_stochastic_move, self.__move(s, Action.left)),
                    (p_stochastic_move, self.__move(s, Action.right))]
                    
        if (a == Action.left) or (a == Action.right):
            return [(p_move, self.__move(s, a)),
                    (p_stochastic_move, self.__move(s, Action.up)),
                    (p_stochastic_move, self.__move(s, Action.down))]
                    
    def reward(self, s):
        return self.env[s[0]][s[1]]
        
    def actions(self, s):
        # Hardcode goal/terminal states for now
        return [Action.none] if ((self.reward(s) == +1) or (self.reward(s) == -1)) else (a for a in Action)
        
    def __move(self, s, a):
        # Account for moves that result in bumping into the wall
        if (a == Action.up):
            if (s[0] == 0) or (self.env[s[0] - 1][s[1]] == None):    # Take care of the blocks
                return s
            return (s[0] - 1, s[1])
        
        if (a == Action.down):
            if (s[0] == self.rows - 1) or (self.env[s[0] + 1][s[1]] == None):
                return s
            return (s[0] + 1, s[1])
            
        if (a == Action.left):
            if (s[1] == 0) or (self.env[s[0]][s[1] - 1] == None):
                return s
            return (s[0], s[1] - 1)
        
        if (a == Action.right):
            if (s[1] == self.cols - 1) or (self.env[s[0]][s[1] + 1] == None):
                return s
            return (s[0], s[1] + 1)

            
# MAIN ------------------------------------------------------------------------
def value_iteration(mdp, e=0.001):
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, g = mdp.reward, mdp.transition, mdp.gamma
    
    while True:
        U = U1.copy()
        delta = 0
        
        for s in mdp.states:
            # Use Bellman-Equation to get the respective utilities
            U1[s] = R(s) + g * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
            
        if delta < e * (1 - g) / g:
             return U

def get_optimal_policy(U, mdp):
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(list(mdp.actions(s)), lambda a:expected_utility(a, s, U, mdp.transition))
    return pi

# Returns expected utility of taking action a in state s, according to the MDP
# and U.
def expected_utility(a, s, U, T):
    return sum([p * U[s1] for (p, s1) in T(s, a)])

# Max action to maximum utility
def argmax(actions, utilityFunc):
    maxAction = actions[0]
    maxUtil = utilityFunc(maxAction)

    for a in actions:
        currentUtil = utilityFunc(a)

        if currentUtil > maxUtil:
            maxUtil = currentUtil
            maxAction = a
    return maxAction

grid = [[-0.04, -0.04, -0.04, -0.04],
        [-0.04, None, -0.04, -1],
        [-0.04, -0.04, -0.04, +1]]

mdp = MDP(grid)
U = value_iteration(mdp)

print("World:")
pprint(grid)
print("\nUtilities:")
pprint(U)
print("\nPolicy:")
pprint(get_optimal_policy(U, mdp))
