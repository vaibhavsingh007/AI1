# This algorithm implements IDDFS
# vsingh38@uic.edu 100317
from enum import Enum
from math import floor
import copy     # For performing deep-copy
import time
import sys      # For capturing memory usage

total_nodes_created = 0     # For memory monitoring purpose

class Moves(Enum):
    none = 0
    up = 1
    down = 2
    left = 3
    right = 4

class PuzzleBoard(object):
    dim = None    # Stores the dimension of the puzzle
    board = []
    blank = None
    last_move = None    # last tile moved
    path = []   # To store the path leading to.
    
    
    def __init__(self, puzzle):
        # initial_puzzle expected to be a 2-D matrix > x0
        self.board = puzzle
        self.dim = len(puzzle[0])
        self.blank = self.__get_blank_position()

    def __get_blank_position(self):
        for i in range(self.dim):
                for j in range(self.dim):
                    if self.board[i][j] == 0:
                        return [i,j]

    def __get_move(self, ti, tj):
        if self.blank[0] > 0 and ti == self.blank[0] - 1 and tj == self.blank[1]:
            return Moves.down
        elif self.blank[0] < self.dim - 1 and ti == self.blank[0] + 1 and tj == self.blank[1]:
            return Moves.up
        elif self.blank[1] > 0 and ti == self.blank[0] and tj == self.blank[1] - 1:
            return Moves.right
        elif self.blank[1] < self.dim - 1 and ti == self.blank[0] and tj == self.blank[1] + 1:
            return Moves.left
        else:
            return Moves.none

    # Swaps the tile with blank
    def swap(self, ti, tj):
        self.board[self.blank[0]][self.blank[1]] = self.board[ti][tj]
        self.board[ti][tj] = 0
        self.blank = [ti,tj]

    def move_tile(self, ti, tj):
        m = self.__get_move(ti, tj)

        if m == None:
            return

        moved_from = Moves.none

        # Last move is used for local pruning.
        self.last_move = self.board[ti][tj]
        self.swap(ti, tj)
        return m

    # Checks if this is the goal state
    def is_goal(self):
        for i in range(self.dim):
            for j in range(self.dim):
                tile = self.board[i][j]
                
                if tile != 0:
                    # Verify if tile is at its designated position
                    goal_row = floor((tile - 1) / self.dim)
                    goal_col = (tile - 1) % self.dim

                    if goal_row != i or goal_col != j:
                        return False
        return True

    def get_possible_moves(self):
        tiles_to_move = []
        
        def __try_append(i, j):
            move = self.__get_move(i, j)
            if move != Moves.none and i >= 0 and i < self.dim and j >= 0 and j < self.dim: 
                tiles_to_move.append([i, j])

        # Tile above
        ti = self.blank[0] - 1
        tj = self.blank[1]
        __try_append(ti,tj)

        # Tile below
        ti = self.blank[0] + 1
        __try_append(ti,tj)

        # Tile left
        ti = self.blank[0]
        tj = self.blank[1] - 1
        __try_append(ti,tj)

        # Tile right
        tj = self.blank[1] + 1
        __try_append(ti,tj)

        return tiles_to_move

    # Deep clones current board
    def clone(self):
        cloned_board = PuzzleBoard(copy.deepcopy(self.board))
        cloned_board.path = copy.deepcopy(self.path)
        return cloned_board

    

    # Expands possible states from current node
    def expand(self):
        children = []
        possible_moves = self.get_possible_moves()
        global total_nodes_created

        for tile in possible_moves:
            if (self.board[tile[0]][tile[1]] != self.last_move):    # Prevent going back
                clone = copy.deepcopy(self)
                clone.move_tile(tile[0], tile[1])   # Note that tile move is implicitly directed towards the blank.
                clone.path.append(tile)
                children.append(clone)
                total_nodes_created += 1
        return children
        
    # Solve using Iterative Deepening Depth First Search
    def solve_using_IDDFS(self):
        root = self.clone()
        t0 = time.time()

        # Set an arbitrary depth limit that may be calibrated
        max_depth = 50

        for d in range(max_depth):
            result = self.__dls(root, d)
            if result != None:
                t1 = time.time()
                print("Time taken: %s s" % (t1 - t0))
                print("Solution found at depth: ", d)
                print("Total Nodes: %s. \nTotal space (appx): %s KB" % (total_nodes_created, ((sys.getsizeof(self) * total_nodes_created)/1024)))
                return result
        return None

    def __dls(self, current_node, max_depth):
        # Corner condition
        if max_depth <= 0:
            return None

        if current_node.is_goal():
            return current_node.path

        # Else, expand node and dive right in
        for node in current_node.expand():
            result = self.__dls(node, max_depth-1)
            if result != None:
                return result
        return None
            
# MAIN
# Create random puzzles (test cases)
#puzzle = [
#    [1,2,3,4],
#    [5,6,8,0],
#    [9,10,7,11],
#    [13,14,15,12],
#    ]
# Ans: [[1, 2], [2, 2], [2, 3], [3, 3]]
puzzle = [
    [1,6,2,3],
    [5,7,11,0],
    [9,10,8,4],
    [13,14,15,12]
    ]
# Ans: [[2, 3], [2, 2], [1, 2], [1, 1], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]]

puzzle_board = PuzzleBoard(puzzle)
solution = puzzle_board.solve_using_IDDFS()

if solution != None:
    print("Solution:")
    print(solution)
else:
    print("No solution found")

# Note that the solution prints the sequence of tile swap (with the blank) using tile indices.
