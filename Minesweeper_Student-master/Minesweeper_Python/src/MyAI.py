# import random
# from collections import deque
#
# from AI import AI
# from Action import Action
#
#
# def count_mix(row):
#     ones = 0
#     neg_ones = 0
#     onesList = []
#     negList = []
#     for _ in range(len(row)):
#         if row[_] == 1:
#             ones += 1
#             onesList.append(_)
#         if row[_] == -1:
#             neg_ones += 1
#             negList.append(_)
#     return ones, onesList, neg_ones, negList
#
#
# class MyAI(AI):
#     class __Tile:
#         mine = False
#         covered = True
#         flag = False
#         number = -100
#
#     class MineRecord:
#         def __init__(self, totalMines):
#             self.mines = []
#             self.mines_left = totalMines
#
#     def __init__(self, rowDimension, colDimension, totalMines, startX, startY):
#         self.shape = [rowDimension, colDimension]
#         self.start_pos = [startX, startY]
#         self.prev_pos = [startX, startY]
#
#         self.safe_q = deque([])
#         self.unknown_tile = [(j, i) for i in range(self.shape[0]) for j in range(self.shape[1])]
#         self.unknown_tile.remove((self.start_pos[0], self.start_pos[1]))
#         self.prob = dict()
#         self.mine_rec = self.MineRecord(totalMines)
#
#         self.board = [[self.__Tile() for _ in range(self.shape[0])] for _ in range(self.shape[1])]
#         self.board[self.start_pos[0]][self.start_pos[1]].covered = False
#         self.board[self.start_pos[0]][self.start_pos[1]].nubmer = 0
#         self.prev_tile = self.board[self.start_pos[0]][self.start_pos[1]]
#         self.prev_action = AI.Action(1)
#
#     def getAction(self, number: int) -> "Action Object":
#         if self.prev_action == AI.Action(1):
#             self.prev_tile.covered = False
#             self.prev_tile.number = number
#
#         # Zero Propagation
#         if number == 0:
#             for col in range(self.prev_pos[0] - 1, self.prev_pos[0] + 2):
#                 for row in range(self.prev_pos[1] - 1, self.prev_pos[1] + 2):
#                     if (self.is_in_bound(col, row) and not (col == self.prev_pos[0] and row == self.prev_pos[1])) and (
#                             (col, row) not in self.safe_q) and self.board[col][row].covered:
#                         self.safe_q.append((col, row))
#
#         while self.safe_q != deque([]):
#             cr = self.safe_q.popleft()
#             self.record_move(AI.Action(1), cr[0], cr[1])
#             return Action(AI.Action(1), cr[0], cr[1])
#
#         # Single Square Constraint 1
#         for col in range(0, self.shape[1]):
#             for row in range(0, self.shape[0]):
#                 if (not self.board[col][row].covered) and self.board[col][row].number != 0 and self.board[col][
#                     row].number == self.neigh_covered(col, row)[0]:
#                     mines = self.neigh_covered(col, row)[1]
#                     for pos in mines:
#                         self.record_mine(pos)
#         # Single Square Constraint 2
#         for col in range(0, self.shape[1]):
#             for row in range(0, self.shape[0]):
#                 if (self.board[col][row].number == self.neigh_mines(col, row)[0]) and (
#                         self.neigh_covered(col, row)[0] - self.neigh_mines(col, row)[0] > 0):
#                     covered = self.neigh_covered(col, row)[1]
#                     mines = self.neigh_mines(col, row)[1]
#                     for pos in covered:
#                         if (pos not in mines) and (pos not in self.safe_q):
#                             self.safe_q.append(pos)
#
#         while self.safe_q != deque([]):
#             cr = self.safe_q.popleft()
#             self.record_move(AI.Action(1), cr[0], cr[1])
#             return Action(AI.Action(1), cr[0], cr[1])
#
#         for col in range(self.shape[1]):
#             for row in range(self.shape[0]):
#                 if self.board[col][row].number > 0 and \
#                         self.neigh_unknown(col, row)[0] > 0:
#                     safe_neigh = self.neighbor_test(col, row)
#                     if safe_neigh is not None and safe_neigh != []:
#                         for pos in safe_neigh:
#                             if pos in self.unknown_tile and pos not in self.safe_q:
#                                 self.safe_q.append(pos)
#
#         while self.safe_q != deque([]):
#             cr = self.safe_q.popleft()
#             self.record_move(AI.Action(1), cr[0], cr[1])
#             return Action(AI.Action(1), cr[0], cr[1])
#
#         unknown = self.unknown_tile
#         total_mines_left = self.mine_rec.mines_left
#         constraints = []
#         for col in range(self.shape[1]):
#             for row in range(self.shape[0]):
#                 if self.board[col][row].number > 0 and self.neigh_unknown(col, row)[0] > 0:
#                     constraints.append((col, row))
#         constraints_cnt = len(constraints)
#         row_cnt = constraints_cnt + 1
#         col_cnt = len(unknown) + 1
#
#         if col_cnt != 1 and row_cnt != 1:
#             columnHeader = [x for x in range(col_cnt)]
#             frontierHeader = columnHeader[:-1]
#             col_to_tile = dict(zip(frontierHeader, unknown))
#             tile_to_col = dict(zip(unknown, frontierHeader))
#
#             # Gaussian Elimination
#             matrix = [[0 for i in range(col_cnt)] for j in range(row_cnt)]
#             row = 0
#             for constraint in constraints:
#                 sub_frontier = self.neigh_unknown(constraint[0], constraint[1])[1]
#                 for tile in sub_frontier:
#                     col = tile_to_col.get(tile)
#                     matrix[row][col] = 1
#                 minesCount = self.board[constraint[0]][constraint[1]].number - \
#                              self.neigh_mines(constraint[0], constraint[1])[0]
#                 matrix[row][-1] = minesCount
#                 row += 1
#             for i in range(col_cnt):
#                 matrix[row][i] = 1
#             matrix[-1][-1] = total_mines_left
#             j = 0
#             column_num = len(matrix[0])
#             for ro in range(len(matrix)):
#                 if j >= column_num:
#                     break
#                 i = ro
#                 flag = False
#                 while matrix[i][j] == 0:
#                     i += 1
#                     if i == len(matrix):
#                         i = ro
#                         j += 1
#                         if column_num == j:
#                             flag = True
#                             break
#                 if flag:
#                     break
#                 matrix[i], matrix[ro] = matrix[ro], matrix[i]
#                 level = matrix[ro][j]
#                 matrix[ro] = [int(_ / level) for _ in matrix[ro]]
#                 for i in range(len(matrix)):
#                     if i != ro:
#                         level = matrix[i][j]
#                         matrix[i] = [iv - level * rv for rv, iv in zip(matrix[ro], matrix[i])]
#                 j += 1
#
#             safe = []
#             mines = []
#             for row in matrix:
#                 last = row[-1]
#                 ones_cnt = count_mix(row[:-1])[0]
#                 ones_list = count_mix(row[:-1])[1]
#                 neg_ones_cnt = count_mix(row[:-1])[2]
#                 neg_list = count_mix(row[:-1])[3]
#
#                 if last == 0:
#                     if ones_cnt > 0 and neg_ones_cnt == 0:
#                         for col in ones_list:
#                             tile = col_to_tile.get(col)
#                             if tile not in safe:
#                                 safe.append(tile)
#                     if neg_ones_cnt > 0 and ones_cnt == 0:
#                         for col in neg_list:
#                             tile = col_to_tile.get(col)
#                             if tile not in mines:
#                                 mines.append(tile)
#                 if last > 0:
#                     if ones_cnt == last:
#                         for col in ones_list:
#                             tile = col_to_tile.get(col)
#                             if tile not in safe:
#                                 mines.append(tile)
#                         for col in neg_list:
#                             tile = col_to_tile.get(col)
#                             if tile not in mines:
#                                 safe.append(tile)
#                 if last < 0:
#                     if neg_ones_cnt == last:
#                         for col in ones_list:
#                             tile = col_to_tile.get(col)
#                             if tile not in safe:
#                                 safe.append(tile)
#                         for col in neg_list:
#                             tile = col_to_tile.get(col)
#                             if tile not in mines:
#                                 mines.append(tile)
#             if mines:
#                 for pos in mines:
#                     self.record_mine(pos)
#
#             if safe:
#                 for pos in safe:
#                     if pos in self.unknown_tile and pos not in self.safe_q:
#                         self.safe_q.append(pos)
#
#         while self.safe_q != deque([]):
#             cr = self.safe_q.popleft()
#             self.record_move(AI.Action(1), cr[0], cr[1])
#             return Action(AI.Action(1), cr[0], cr[1])
#
#         # Probability Guessing
#         if self.unknown_tile:
#             keys = self.unknown_tile
#             values = [self.mine_rec.mines_left / len(self.unknown_tile)] * len(self.unknown_tile)
#             self.prob = dict(zip(keys, values))
#         for col in range(0, self.shape[1]):
#             for row in range(0, self.shape[0]):
#                 num_mines = self.neigh_mines(col, row)[0]
#                 num_covered = self.neigh_covered(col, row)[0]
#                 if (self.board[col][row].number > 0) and (num_covered - num_mines > 0):
#                     mines = self.neigh_mines(col, row)[1]
#                     covered = self.neigh_covered(col, row)[1]
#                     for pos in covered:
#                         if (pos not in mines) and (pos not in self.safe_q):
#                             self.prob[pos] = max((self.board[col][row].number - num_mines) / num_covered,
#                                                  self.prob[pos])
#
#         for pos in [(self.shape[1] - 1, self.shape[0] - 1), (0, 0), (self.shape[1] - 1, 0), (0, self.shape[0] - 1)]:
#             if pos in self.unknown_tile:
#                 self.prob[pos] = self.prob[pos] - 1
#
#         if self.unknown_tile:
#             min_list = []
#             min = float("inf")
#             for k, v in self.prob.items():
#                 if v == min:
#                     min_list.append(k)
#                 if v < min:
#                     min = v
#                     min_list = [k]
#             self.safe_q.append(random.choice(min_list))
#         if self.mine_rec.mines_left == 0:
#             return Action(AI.Action(0))
#
#         while self.safe_q != deque([]):
#             cr = self.safe_q.popleft()
#             self.record_move(AI.Action(1), cr[0], cr[1])
#             return Action(AI.Action(1), cr[0], cr[1])
#
#         if self.mine_rec.mines_left == 0:
#             return Action(AI.Action(0))
#
#     def record_mine(self, pos):
#         if (pos[0], pos[1]) not in self.mine_rec.mines:
#             self.mine_rec.mines_left -= 1
#             self.mine_rec.mines.append((pos[0], pos[1]))
#             self.board[pos[0]][pos[1]].mine = True
#             self.board[pos[0]][pos[1]].flag = True
#             self.unknown_tile.remove((pos[0], pos[1]))
#
#     def record_move(self, action, c, r):
#         self.prev_pos[0] = c
#         self.prev_pos[1] = r
#         self.prev_tile = self.board[c][r]
#         self.prev_action = action
#         self.unknown_tile.remove((c, r))
#         if (c, r) in list(self.prob.keys()):
#             self.prob.pop((c, r))
#
#     def is_in_bound(self, c: int, r: int) -> bool:
#         return self.shape[1] > c >= 0 and self.shape[0] > r >= 0
#
#     def neigh_tile(self, pos):
#         tiles = set()
#         for c in range(pos[0] - 1, pos[0] + 2):
#             for r in range(pos[1] - 1, pos[1] + 2):
#                 if self.is_in_bound(c, r) and (c, r) != pos:
#                     tiles.add((c, r))
#         return tiles
#
#     def neigh_unknown(self, col, row):
#         count = 0
#         no_flag = []
#         for c in range(col - 1, col + 2):
#             for r in range(row - 1, row + 2):
#                 if self.is_in_bound(c, r) and (c, r) != (col, row):
#                     if not self.is_known((c, r)):
#                         count += 1
#                         no_flag.append((c, r))
#         return count, no_flag
#
#     def neigh_covered(self, col, row):
#         count = 0
#         covered = []
#         for c in range(col - 1, col + 2):
#             for r in range(row - 1, row + 2):
#                 if self.is_in_bound(c, r) and (c, r) != (col, row):
#                     if self.board[c][r].covered:
#                         count += 1
#                         covered.append((c, r))
#         return count, covered
#
#     def neigh_mines(self, col, row):
#         count = 0
#         s_mines = []
#         for c in range(col - 1, col + 2):
#             for r in range(row - 1, row + 2):
#                 if self.is_in_bound(c, r):
#                     if self.board[c][r].mine:
#                         self.board[c][r].flag = True
#                         count += 1
#                         s_mines.append((c, r))
#         return count, s_mines
#
#     def set_known(self, set):
#         for i in set:
#             if not self.is_known(i):
#                 return False
#         return True
#
#     def is_known(self, coord):
#         return not (self.board[coord[0]][coord[1]].covered and (not self.board[coord[0]][coord[1]].flag))
#
#     # Multi Square Constraint
#     def neighbor_test(self, col, row):
#         safe_neigh = []
#         center = (col, row)
#         percept_center = self.board[col][row].number
#         neighbors_list = []
#         for co in range(col - 2, col + 3):
#             for ro in range(row - 2, row + 3):
#                 if self.is_in_bound(co, ro) and (co, ro) != (col, row):
#                     neighbors_list.append((co, ro))
#         for neighbor in set(neighbors_list):
#             percept_neighbor = self.board[neighbor[0]][neighbor[1]].number
#             if percept_neighbor >= 1:
#                 N = self.neigh_tile(neighbor)
#                 if center in N:
#                     N.remove(center)
#
#                 if not self.set_known(N):
#                     A = self.neigh_tile(center)
#                     if neighbor in A:
#                         A.remove(neighbor)
#
#                     N_not_A = N.difference(A)
#                     A_not_N = A.difference(N)
#
#                     mines_A = set(self.neigh_mines(center[0], center[1])[1])
#                     mines_N = set(self.neigh_mines(neighbor[0], neighbor[1])[1])
#
#                     mines_both = mines_A.intersection(N)
#                     mines_A_not_N = mines_A.intersection(A_not_N)
#                     mines_N_not_A = mines_N.intersection(N_not_A)
#
#                     if self.set_known(N_not_A):
#                         if len(mines_N_not_A) == 0 and len(mines_both) == 0:
#                             if (percept_center - len(mines_A_not_N)) == (percept_neighbor - len(mines_N_not_A)):
#                                 for coord in A_not_N:
#                                     if not self.is_known(coord):
#                                         safe_neigh.append(coord)
#         return safe_neigh

from collections import deque
import random

from AI import AI
from Action import Action

def count_mix(row):
    """
    Count and track positions of 1s and -1s in a row
    Returns: (count of 1s, positions of 1s, count of -1s, positions of -1s)
    """
    ones = 0
    neg_ones = 0
    ones_list = []
    neg_list = []
    for i in range(len(row)):
        if row[i] == 1:
            ones += 1
            ones_list.append(i)
        if row[i] == -1:
            neg_ones += 1
            neg_list.append(i)
    return ones, ones_list, neg_ones, neg_list

class MyAI(AI):
    class __Tile:
        mine = False
        covered = True
        flag = False
        number = -100

    class MineRecord:
        def __init__(self, totalMines):
            self.mines = []
            self.mines_left = totalMines

    def __init__(self, rowDimension, colDimension, totalMines, startX, startY):
        """Initialize the AI with board dimensions and starting position"""
        self.shape = [rowDimension, colDimension]
        self.start_pos = [startX, startY]
        self.prev_pos = [startX, startY]

        # Initialize tracking queues and sets
        self.safe_q = deque([])  # Queue of safe moves
        self.unknown_tile = [(j, i) for i in range(self.shape[0]) for j in range(self.shape[1])]
        self.unknown_tile.remove((self.start_pos[0], self.start_pos[1]))
        self.prob = dict()  # Store probability calculations
        self.mine_rec = self.MineRecord(totalMines)

        # Initialize board state
        self.board = [[self.__Tile() for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        self.board[self.start_pos[0]][self.start_pos[1]].covered = False
        self.board[self.start_pos[0]][self.start_pos[1]].number = 0
        self.prev_tile = self.board[self.start_pos[0]][self.start_pos[1]]
        self.prev_action = AI.Action(1)

    def getAction(self, number: int) -> "Action Object":
        """
        Determine next move based on current board state and previous number revealed
        Uses multiple strategies in order:
        1. Zero propagation
        2. Single square constraints
        3. Multi-square constraints
        4. Gaussian elimination
        5. Probability-based guessing
        """
        # Update state from previous move
        if self.prev_action == AI.Action(1):
            self.prev_tile.covered = False
            self.prev_tile.number = number

        # Strategy 1: Zero Propagation
        if number == 0:
            for col in range(self.prev_pos[0] - 1, self.prev_pos[0] + 2):
                for row in range(self.prev_pos[1] - 1, self.prev_pos[1] + 2):
                    if (self.is_in_bound(col, row) and not (col == self.prev_pos[0] and row == self.prev_pos[1])) and (
                            (col, row) not in self.safe_q) and self.board[col][row].covered:
                        self.safe_q.append((col, row))

        # Process safe moves queue
        while self.safe_q:
            cr = self.safe_q.popleft()
            self.record_move(AI.Action(1), cr[0], cr[1])
            return Action(AI.Action(1), cr[0], cr[1])

        # Strategy 2: Single Square Constraint 1 - Mark definite mines
        for col in range(0, self.shape[1]):
            for row in range(0, self.shape[0]):
                if (not self.board[col][row].covered) and self.board[col][row].number != 0 and \
                        self.board[col][row].number == self.neigh_covered(col, row)[0]:
                    mines = self.neigh_covered(col, row)[1]
                    for pos in mines:
                        self.record_mine(pos)

        # Strategy 2: Single Square Constraint 2 - Mark definite safe tiles
        for col in range(0, self.shape[1]):
            for row in range(0, self.shape[0]):
                if (self.board[col][row].number == self.neigh_mines(col, row)[0]) and (
                        self.neigh_covered(col, row)[0] - self.neigh_mines(col, row)[0] > 0):
                    covered = self.neigh_covered(col, row)[1]
                    mines = self.neigh_mines(col, row)[1]
                    for pos in covered:
                        if (pos not in mines) and (pos not in self.safe_q):
                            self.safe_q.append(pos)

        while self.safe_q:
            cr = self.safe_q.popleft()
            self.record_move(AI.Action(1), cr[0], cr[1])
            return Action(AI.Action(1), cr[0], cr[1])

        # Strategy 3: Multi-square Constraint Analysis
        for col in range(self.shape[1]):
            for row in range(self.shape[0]):
                if self.board[col][row].number > 0 and \
                        self.neigh_unknown(col, row)[0] > 0:
                    safe_neigh = self.neighbor_test(col, row)
                    if safe_neigh is not None and safe_neigh:
                        for pos in safe_neigh:
                            if pos in self.unknown_tile and pos not in self.safe_q:
                                self.safe_q.append(pos)

        while self.safe_q:
            cr = self.safe_q.popleft()
            self.record_move(AI.Action(1), cr[0], cr[1])
            return Action(AI.Action(1), cr[0], cr[1])

        # Strategy 4: Gaussian Elimination for Complex Constraints
        unknown = self.unknown_tile
        total_mines_left = self.mine_rec.mines_left
        constraints = []
        for col in range(self.shape[1]):
            for row in range(self.shape[0]):
                if self.board[col][row].number > 0 and self.neigh_unknown(col, row)[0] > 0:
                    constraints.append((col, row))
        constraints_cnt = len(constraints)
        row_cnt = constraints_cnt + 1
        col_cnt = len(unknown) + 1

        if col_cnt != 1 and row_cnt != 1:
            columnHeader = [x for x in range(col_cnt)]
            frontierHeader = columnHeader[:-1]
            col_to_tile = dict(zip(frontierHeader, unknown))
            tile_to_col = dict(zip(unknown, frontierHeader))

            # Build and solve constraint matrix
            matrix = [[0 for i in range(col_cnt)] for j in range(row_cnt)]
            row = 0
            for constraint in constraints:
                sub_frontier = self.neigh_unknown(constraint[0], constraint[1])[1]
                for tile in sub_frontier:
                    col = tile_to_col.get(tile)
                    matrix[row][col] = 1
                minesCount = self.board[constraint[0]][constraint[1]].number - \
                             self.neigh_mines(constraint[0], constraint[1])[0]
                matrix[row][-1] = minesCount
                row += 1
            for i in range(col_cnt):
                matrix[row][i] = 1
            matrix[-1][-1] = total_mines_left

            # Perform Gaussian elimination
            j = 0
            column_num = len(matrix[0])
            for ro in range(len(matrix)):
                if j >= column_num:
                    break
                i = ro
                flag = False
                while matrix[i][j] == 0:
                    i += 1
                    if i == len(matrix):
                        i = ro
                        j += 1
                        if column_num == j:
                            flag = True
                            break
                if flag:
                    break
                matrix[i], matrix[ro] = matrix[ro], matrix[i]
                level = matrix[ro][j]
                matrix[ro] = [int(_ / level) for _ in matrix[ro]]
                for i in range(len(matrix)):
                    if i != ro:
                        level = matrix[i][j]
                        matrix[i] = [iv - level * rv for rv, iv in zip(matrix[ro], matrix[i])]
                j += 1

            # Process solution
            safe = []
            mines = []
            for row in matrix:
                last = row[-1]
                ones_cnt, ones_list, neg_ones_cnt, neg_list = count_mix(row[:-1])

                if last == 0:
                    if ones_cnt > 0 and neg_ones_cnt == 0:
                        for col in ones_list:
                            tile = col_to_tile.get(col)
                            if tile not in safe:
                                safe.append(tile)
                    if neg_ones_cnt > 0 and ones_cnt == 0:
                        for col in neg_list:
                            tile = col_to_tile.get(col)
                            if tile not in mines:
                                mines.append(tile)
                if last > 0:
                    if ones_cnt == last:
                        for col in ones_list:
                            tile = col_to_tile.get(col)
                            if tile not in safe:
                                mines.append(tile)
                        for col in neg_list:
                            tile = col_to_tile.get(col)
                            if tile not in mines:
                                safe.append(tile)
                if last < 0:
                    if neg_ones_cnt == last:
                        for col in ones_list:
                            tile = col_to_tile.get(col)
                            if tile not in safe:
                                safe.append(tile)
                        for col in neg_list:
                            tile = col_to_tile.get(col)
                            if tile not in mines:
                                mines.append(tile)

            # Record found mines and safe tiles
            if mines:
                for pos in mines:
                    self.record_mine(pos)

            if safe:
                for pos in safe:
                    if pos in self.unknown_tile and pos not in self.safe_q:
                        self.safe_q.append(pos)

        while self.safe_q:
            cr = self.safe_q.popleft()
            self.record_move(AI.Action(1), cr[0], cr[1])
            return Action(AI.Action(1), cr[0], cr[1])

        # Strategy 5: Probability-based Guessing
        if self.unknown_tile:
            # Initialize base probabilities
            keys = self.unknown_tile
            values = [self.mine_rec.mines_left / len(self.unknown_tile)] * len(self.unknown_tile)
            self.prob = dict(zip(keys, values))

            # Adjust probabilities based on neighbors
            for col in range(0, self.shape[1]):
                for row in range(0, self.shape[0]):
                    num_mines = self.neigh_mines(col, row)[0]
                    num_covered = self.neigh_covered(col, row)[0]
                    if (self.board[col][row].number > 0) and (num_covered - num_mines > 0):
                        mines = self.neigh_mines(col, row)[1]
                        covered = self.neigh_covered(col, row)[1]
                        for pos in covered:
                            if (pos not in mines) and (pos not in self.safe_q):
                                self.prob[pos] = max((self.board[col][row].number - num_mines) / num_covered,
                                                     self.prob[pos])

            # Adjust corner probabilities
            for pos in [(self.shape[1] - 1, self.shape[0] - 1), (0, 0),
                       (self.shape[1] - 1, 0), (0, self.shape[0] - 1)]:
                if pos in self.unknown_tile:
                    self.prob[pos] = self.prob[pos] - 1

            # Choose safest move
            if self.unknown_tile:
                min_list = []
                min_val = float("inf")
                for k, v in self.prob.items():
                    if v == min_val:
                        min_list.append(k)
                    if v < min_val:
                        min_val = v
                        min_list = [k]
                self.safe_q.append(random.choice(min_list))

        if self.mine_rec.mines_left == 0:
            return Action(AI.Action(0))

        while self.safe_q:
            cr = self.safe_q.popleft()
            self.record_move(AI.Action(1), cr[0], cr[1])
            return Action(AI.Action(1), cr[0], cr[1])

        if self.mine_rec.mines_left == 0:
            return Action(AI.Action(0))

    def record_mine(self, pos):
        """Record a discovered mine position"""
        if (pos[0], pos[1]) not in self.mine_rec.mines:
            self.mine_rec.mines_left -= 1
            self.mine_rec.mines.append((pos[0], pos[1]))
            self.board[pos[0]][pos[1]].mine = True
            self.board[pos[0]][pos[1]].flag = True
            self.unknown_tile.remove((pos[0], pos[1]))

    def record_move(self, action, c, r):
        """Record a move made by the AI"""
        self.prev_pos[0] = c
        self.prev_pos[1] = r
        self.prev_tile = self.board[c][r]
        self.prev_action = action
        self.unknown_tile.remove((c, r))
        if (c, r) in list(self.prob.keys()):
            self.prob.pop((c, r))

    def is_in_bound(self, c: int, r: int) -> bool:
        """Check if position is within board boundaries"""
        return self.shape[1] > c >= 0 and self.shape[0] > r >= 0

    def neigh_tile(self, pos):
        """Get set of valid neighboring tile positions"""
        tiles = set()
        for c in range(pos[0] - 1, pos[0] + 2):
            for r in range(pos[1] - 1, pos[1] + 2):
                if self.is_in_bound(c, r) and (c, r) != pos:
                    tiles.add((c, r))
        return tiles

    def neigh_unknown(self, col, row):
        """Get count and positions of unknown neighboring tiles"""
        count = 0
        no_flag = []
        for c in range(col - 1, col + 2):
            for r in range(row - 1, row + 2):
                if self.is_in_bound(c, r) and (c, r) != (col, row):
                    if not self.is_known((c, r)):
                        count += 1
                        no_flag.append((c, r))
        return count, no_flag

    def neigh_covered(self, col, row):
        """Get count and positions of covered neighboring tiles"""
        count = 0
        covered = []
        for c in range(col - 1, col + 2):
            for r in range(row - 1, row + 2):
                if self.is_in_bound(c, r) and (c, r) != (col, row):
                    if self.board[c][r].covered:
                        count += 1
                        covered.append((c, r))
        return count, covered

    def neigh_mines(self, col, row):
        """Get count and positions of identified neighboring mines"""
        count = 0
        s_mines = []
        for c in range(col - 1, col + 2):
            for r in range(row - 1, row + 2):
                if self.is_in_bound(c, r):
                    if self.board[c][r].mine:
                        self.board[c][r].flag = True
                        count += 1
                        s_mines.append((c, r))
        return count, s_mines

    def set_known(self, set):
        """Check if all tiles in a set have been explored"""
        for i in set:
            if not self.is_known(i):
                return False
        return True

    def is_known(self, coord):
        """Check if a tile's content is known (either uncovered or flagged)"""
        return not (self.board[coord[0]][coord[1]].covered and (not self.board[coord[0]][coord[1]].flag))

    def neighbor_test(self, col, row):
        """
        Implement multi-square constraint analysis
        Analyzes patterns between neighboring number tiles to deduce safe moves
        """
        safe_neigh = []
        center = (col, row)
        percept_center = self.board[col][row].number
        neighbors_list = []

        # Get extended neighborhood (2 tiles out)
        for co in range(col - 2, col + 3):
            for ro in range(row - 2, row + 3):
                if self.is_in_bound(co, ro) and (co, ro) != (col, row):
                    neighbors_list.append((co, ro))

        for neighbor in set(neighbors_list):
            percept_neighbor = self.board[neighbor[0]][neighbor[1]].number
            if percept_neighbor >= 1:
                N = self.neigh_tile(neighbor)
                if center in N:
                    N.remove(center)

                if not self.set_known(N):
                    A = self.neigh_tile(center)
                    if neighbor in A:
                        A.remove(neighbor)

                    # Calculate set differences
                    N_not_A = N.difference(A)
                    A_not_N = A.difference(N)

                    # Get mine positions
                    mines_A = set(self.neigh_mines(center[0], center[1])[1])
                    mines_N = set(self.neigh_mines(neighbor[0], neighbor[1])[1])

                    # Calculate intersection and differences for mines
                    mines_both = mines_A.intersection(N)
                    mines_A_not_N = mines_A.intersection(A_not_N)
                    mines_N_not_A = mines_N.intersection(N_not_A)

                    # Apply advanced pattern recognition
                    if self.set_known(N_not_A):
                        if len(mines_N_not_A) == 0 and len(mines_both) == 0:
                            if (percept_center - len(mines_A_not_N)) == (percept_neighbor - len(mines_N_not_A)):
                                for coord in A_not_N:
                                    if not self.is_known(coord):
                                        safe_neigh.append(coord)
        return safe_neigh
