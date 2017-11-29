# This program solves the 15-Puzzle using A* Search.
# Note that two heuristic functions are inclued, namely, 
#..Misplaced Tiles and Manhattan Distance.
# The heuristic function used can be switched in the h() function call.
# Running time of the algorithm is O(E.lov(V)) where E is |Edges|
#..and V is |V| in the Search Tree.
# vsingh38@uic.edu 100817
from enum import Enum
from math import floor, fabs
import copy
import time
import sys      # For capturing memory usage
import heapq

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
    # Note that tile move is implicitly directed towards the blank.
    
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

        for tile in possible_moves:
            if (self.board[tile[0]][tile[1]] != self.last_move):    # Prevent going back
                clone = copy.deepcopy(self)
                clone.move_tile(tile[0], tile[1])
                clone.path.append(tile)
                children.append(clone)
        return children

    # Heuristics
    def get_misplaced_tiles(self):
        count = 0

        for i in range(self.dim):
            for j in range(self.dim):
                tile = self.board[i][j]
                
                if tile != 0:
                    # Verify if tile is at its designated position
                    goal_row = floor((tile - 1) / self.dim)
                    goal_col = (tile - 1) % self.dim

                    if goal_row != i or goal_col != j:
                        count += 1
        return count

    def get_manhattan_distance(self):
        distance = 0

        for i in range(self.dim):
            for j in range(self.dim):
                tile = self.board[i][j]
                
                if tile != 0:
                    # Verify if tile is at its designated position
                    goal_row = floor((tile - 1) / self.dim)
                    goal_col = (tile - 1) % self.dim

                    if goal_row != i or goal_col != j:
                        distance +=  fabs((i - goal_row) + (j - goal_col))
        return distance

    def __h(self):
        return self.get_misplaced_tiles()   # Switch with Manhattan method when required.

    def __g(self):
        return len(self.path)
    
    # O(E.log(V))
    def solve_using_Astar(self):
        root = self.clone()
        frontier = []
        node_count = 0
        entry_count = 0  # Heap entry counter
        t0 = time.time()

        # Initialize min-heap
        # Heap operation = O(log n)
        entry = [0, entry_count, root]   
        # The above line is to handle tie-breaker issue described in sec 8.4.2, 
        # ..here:https://docs.python.org/2/library/heapq.html
        heapq.heappush(frontier, entry)

        while len(frontier) > 0:
            current_node = heapq.heappop(frontier)[2]
            node_count += 1

            if current_node.is_goal():
                t1 = time.time()
                print("Time taken: %s s" % (t1 - t0))
                print("Nodes visited: %s" % node_count)
                print("Latest list size: %s KB" % (sys.getsizeof(frontier) / 1024))
                return current_node.path

            # Update heap values for child nodes.
            for child in current_node.expand():
                entry_count += 1
                entry = [child.__g() + child.__h(), entry_count, child]
                heapq.heappush(frontier, entry)

            #self.__print_frontier(frontier)
        return None

    # For debugging and analysis purposes
    def __print_frontier(self, frontier):
        for node in frontier:
            for row in node.board:
                print(row)
            print("--")
        input("Hit enter..")

# MAIN
# Create random puzzles (test cases)
#puzzle = [
#    [1,2,3,4],
#    [5,6,8,0],
#    [9,10,7,11],
#    [13,14,15,12],
#    ]
# Ans: [[1, 2], [2, 2], [2, 3], [3, 3]]
puzzle = [[1,6,2,3],
    [5,7,11,0],
    [9,10,8,4],
    [13,14,15,12]]
# Ans: [[2, 3], [2, 2], [1, 2], [1, 1], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]]
"""
Output for test-case 2 (using A-star with Misplaced Tiles heuristics):
----------------------
Time taken: 0.0 s
Nodes visited: 23
Latest list size: 0.2578125 KB
Solution:
[[2, 3], [2, 2], [1, 2], [1, 1], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]]
Press any key to continue . . .

Output for test-case 2 (using BFS):
----------------------
Time taken: 0.4758732318878174 s
Nodes visited: 3362
Latest list size: 30.765625 KB
Solution:
[[2, 3], [2, 2], [1, 2], [1, 1], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]]
Press any key to continue . . .

Output for test-case 2 (using IDDFS):
----------------------
Time taken: 0.7942681312561035 s
Solution found at depth:  11
Total Nodes: 13483.
Total space (appx): 737.3515625 KB
Solution:
[[2, 3], [2, 2], [1, 2], [1, 1], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]]
Press any key to continue . . .
"""

puzzle_board = PuzzleBoard(puzzle)
solution = puzzle_board.solve_using_Astar()

if solution != None:
    print("Solution:")
    print(solution)
else:
    print("No solution found")

# Note that the solution prints the sequence of tile shift (with the blank) using indices.