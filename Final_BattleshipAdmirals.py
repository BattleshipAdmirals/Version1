''' Battleship Admirals 
    AI Agents for playing the game of Battleship using Hasbro boardgame rules
    AI 801 - Fall 2024 - Group 8
'''
import copy
import csv
import math
import random
from IPython.display import clear_output
import time
import numpy as np
import matplotlib.pyplot as plt

''' Game Helper Functions '''
def bitstring_to_grid(bitstring: str):
    size = len(bitstring)
    validate_size(size)
    side_length = int(math.sqrt(size))
    
    grid = [
        ['.' if bitstring[i * side_length + j] == '1' else 'O' for j in range(side_length)]
        for i in range(side_length)
    ]
    return grid

def grid_to_bitstring(grid: list):
    bitstring = ''.join('1' if cell == '.' else '0' for row in grid for cell in row)
    return bitstring

def grid_to_lists(grid: list):
    gridlists = []
    for i in range(len(grid)):
        rowarray = []
        for j in range(len(grid[i])):
            rowvalue = grid[i][j]
            rowarray.append(rowvalue)
        gridlists.append(rowarray)

    for i in range(len(grid)):
        colarray = []
        for j in range(len(grid[i])):
            colvalue = grid[j][i]
            colarray.append(colvalue)
        gridlists.append(colarray)

    return gridlists  

def generate_hidden_grid(grid: list):
    hidden_grid = []
    for i in range(len(grid)):
        rowarray = []
        for j in range(len(grid[i])):
            if grid[i][j] == '*':
                rowarray.append('*')
            elif grid[i][j] == 'O':
                rowarray.append(' ')
            elif grid[i][j] == 'X':
                rowarray.append('X')
            else:
                rowarray.append('?')
        hidden_grid.append(rowarray)
    return hidden_grid

def bitstring_to_int(bitstring: str):
    return int(bitstring,2)

def int_to_bitstring(value: int, length=100):
    bitstring = f'{value:0{length}b}'
    return bitstring

def blank_bitstring(size=100):
    validate_size(size)
    
    return '1' * size

def random_bit():
    return random.choice([0,1])

def random_int(max: int):
    return random.randint(0,max)

def random_location(ship_size: int, grid_size: int):
    return random_int(grid_size-ship_size)
    
def validate_size(size: int):
    side_length = int(math.sqrt(size))
    if side_length * side_length != size:
        raise ValueError("Size must create perfect square")
		
''' Charts and Graphs '''
def chart_results(sorted_agents, title):
    results = []
    labels = []
    for agent in sorted_agents:
        labels.append(agent.name)
        results.append(agent.average)
    plt.barh(labels, results)

    print(title)
    plt.show()

def record_results(agentslist, game_count, filename, mode='w'):
    with open(filename, mode, newline='') as csvfile:  
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Agent", "Game Played", "Average Shots to Win", "Best Game", "Worst Game", "Average First Sunk"])
        for agent in agentslist:
            writer.writerow([agent.name, game_count, agent.average, agent.best, agent.worst, agent.avg_first_sunk])
            
''' Game Grid and Status Display Functions '''
def print_game(grids, shots, hits, hidden, title):
    print("                Battleship Admirals")
    print(title)
    print("        Player 1                  Player 2")
    print_dual_grids(grids[0], grids[1])
    print(f'   Shots: {shots[0]}   Hits: {hits[0]}        Shots: {shots[1]}   Hits: {hits[1]}')
    print()
    print_dual_grids(generate_hidden_grid(grids[0]) if hidden[0] == True else grids[2]
                     , generate_hidden_grid(grids[1]) if hidden[1] == True else grids[3])
    
    print()
    
def print_grid(grid: list):
    row_labels = 'A B C D E F G H I J K L M N'.split()
    print ('   ' + ' '.join([f'{i}' for i in range(len(grid))]))
    
    for i, row in enumerate(grid):
        print(f'{row_labels[i]:2}', end=' ')
        for cell in row:
            print(cell, end=' ')
        print()

def print_dual_grids(grid1: list, grid2: list):
    row_labels = 'A B C D E F G H I J K L M N'.split()
    print ('   ' + ' '.join([f'{i}' for i in range(len(grid1))]) + '       ' + ' '.join([f'{i}' for i in range(len(grid2))]))
    
    for i in range(len(grid1)):
        print(f'{row_labels[i]:2}', end=' ')
        for cell in grid1[i]:
            print(cell, end=' ')

        print('   ', end= '')
        
        print(f'{row_labels[i]:2}', end=' ')
        for cell in grid2[i]:
            print(cell, end=' ')
            
        print()

''' Game Setup and Main Loop '''
#edited to add sunk ships
def random_ship_placement(ship_size: int, grid: list, ships_dict: dict):
    clear = 0
    ship_list = [] #list of list pairs
    while clear == 0:
        orientation = random_bit()
        row = random_location(ship_size, len(grid))
        column = random_location(ship_size, len(grid[0]))
    
        if orientation == 0:
            for i in range(ship_size):
                if grid[row+i][column] == '.':
                    clear = 1
                else:
                    clear = 0
                    break
                
            if clear == 1:
                for i in range(ship_size):
                    grid[row+i][column] = ship_size
                    ship_list.append([(row+i,column),1])
            else:
                continue
        else:
            for i in range(ship_size):
                if grid[row][column+i] == '.':
                    clear = 1
                else:
                    clear = 0
                    break
            if clear == 1:
                for i in range(ship_size):
                    grid[row][column+i] = ship_size
                    ship_list.append([(row,column+i),1])
            else:
                continue 

    ships_dict.append([ship_list, 1])
    return grid, ships_dict

def setup_game(grid_size: int, ships: list):
    bit_string = blank_bitstring(grid_size*grid_size)
    ship_grid1 = bitstring_to_grid(bit_string)
    ship_grid2 = bitstring_to_grid(bit_string)
    shot_grid1 = bitstring_to_grid(bit_string)
    shot_grid2 = bitstring_to_grid(bit_string)
    ships_dict1, ships_dict2 = [], []
    
    for i in range(len(ships)):
        ship_grid1, ships_dict1 = random_ship_placement(ships[i], ship_grid1, ships_dict1)

    for i in range(len(ships)):
        ship_grid2, ships_dict2 = random_ship_placement(ships[i], ship_grid2, ships_dict2)
        
    return shot_grid1, shot_grid2, ship_grid1, ship_grid2, ships_dict1, ships_dict2

def play(grid_size, ships, agents, hidden=(False, False), title="", printgrids=True):
    grid1, grid2, grid3, grid4, dict1, dict2= setup_game(grid_size, ships)
    grids = [grid1,grid2,grid3,grid4]
    sunk = [dict1,dict2]

    hits_to_win = sum(ships)
    max_shots = grid_size*grid_size
    shots = [0, 0]
    hits = [0, 0]
    turn = 0
    first = [0, 0]
    
    if printgrids == True:
        print_game(grids, shots, hits, hidden, title)
    
    while shots[0] < max_shots and shots[1] < max_shots and hits[0] < hits_to_win and hits[1] < hits_to_win:
        if printgrids == True:
            clear_output(wait=True)
        current_shot = agents[turn](grids, turn, ships)

        if grids[turn+2][current_shot[0]][current_shot[1]] == '.':
            grids[turn][current_shot[0]][current_shot[1]] = 'O'
        else:
            grids[turn][current_shot[0]][current_shot[1]] = 'X'
            hits[turn] = hits[turn] + 1
            for i in sunk[turn]: #list of player boats
                index1 = sunk[turn].index(i) #the particular boat
                for j in i[0]: #list of just the tiles
                    index2 = i[0].index(j) #the particular tile
                    if j[0] == (current_shot[0],current_shot[1]):
                        sunk[turn][index1][0][index2][1] = 0
                        is_sunk=0
                        for k in sunk[turn][index1][0]:
                            is_sunk+=k[1]
                        if is_sunk == 0:
                            sunk[turn][index1][1] = 0
                if i[1] == 0:
                    for l in i[0]:
                        grids[turn][l[0][0]][l[0][1]] = '*'
                        if first[turn] == 0:
                            first[turn] = shots[turn]+1
        shots[turn] = shots[turn] + 1
        if turn == 0:
            turn = 1
        else:
            turn = 0

        if printgrids == True:
            clear_output(wait=True)
            print_game(grids, shots, hits, hidden, title)
        if hits[0] >= hits_to_win:
            #print('Player 1 Wins!!!', str(shots[0]) + " shots")
            return np.array([1,0, shots[0], first[0]])
        if hits[1] >= hits_to_win:
            #print('Player 2 Wins!!!', str(shots[1]) + " shots")
            return np.array([0,1, shots[1], first[1]])

''' Human Player Agent: (No AI Agent)
    Asks for and validates the next shot location from the user 
    Allows user to play other users or any of the ai agents
'''
def Human_Player(grids, turn, ships):

    def play_turn(grids, turn):
        valid = 0
        row = -1
        column = -1
        while valid == 0:
            move = input('Enter Valid Row and Column (ex. A9): ')
            row, column = validate_move(grids[turn],move)
            if row >= 0:
                valid = 1

        return row, column

    def validate_move(grid, move):
        row = -1
        column = -1
        try:
            row_labels = list('ABCDEFGHIJKLMN')
            row_value = move[:1].upper()
            column_value = int(move[1:])

            for i in range(len(grid)):
                if row_value == row_labels[i] and column_value >= 0 and column_value < len(grid[0]):
                    row = i
                    column = column_value

            if grid[row][column] != '.':
                row = -1
                column = -1
                print('Duplicate Move, Please try again.')
        except:
            row = -1
            column = -1
            print('Invalid Input, Please try again.')
            
        return row, column
            

    return play_turn(grids, turn)
	
''' Random Chance Agent:
    Randomly picks a valid location for the next shot. 
    No other logic added, purely random chance.
'''
def Random_Agent(grids, turn, ships):
    
    def play_turn(grids, turn):
        clear = 0
        while clear ==0:
            row = random_int(len(grids[turn])-1)
            column = random_int(len(grids[turn][0])-1)
            #print("trying: ", turn , row, column)
            if grids[turn][row][column] == '.':
                clear = 1
        return row, column

    return play_turn(grids, turn)

''' Random Chance Agent Improved:
    Randomly picks a valid location for next shot
    Improved by removing locations surrounded by Misses from possible selected locations
    No ship fits in 1 square
    Example of adding programmed rules to improve decisions even from random chance agents
'''
def Improved_Random_Agent(grids, turn, ships):
    
    def play_turn(grids, turn):
        clear = 0
        long_search = 0
        while clear == 0:
            row = random_int(len(grids[turn])-1)
            column = random_int(len(grids[turn][0])-1)
            clear = check_neighbors(grids[turn], row, column)
            long_search = long_search+1
            if (long_search > 1000):
                #print('LONG SEARCH', turn, row, column)
                time.sleep(5)
        return row, column

    def check_neighbors(grid, row, column):
        row_len = len(grid)
        column_len = len(grid[row])
        
        if grid[row][column] == '.':
            if (row == 0 or (row > 0 and grid[row-1][column] != 'O')) \
                or (row == row_len-1 or (row < row_len-1 and grid[row+1][column] != 'O')) \
                or (column == 0 or (column > 0 and grid[row][column-1] != 'O')) \
                or (column == column_len-1 or (column < column_len-1 and grid[row][column+1] != 'O')):
                return 1
        return 0

    return play_turn(grids, turn)

''' Agent class for Markov Decision Process Agent '''
class Battleship_MDP:
    def __init__ (self,ships, reward, iters):
        # boards is a pair of n*n lists with 0 guess board, and 1 as true
        # on guess board, . is unknown, X is hit, * is sunk, 0 is miss
        # on hit board, . is no ship, otherwise ship
        self.guess_board=[]
        self.size=0
        self.ships=ships # list of ship sizes
        self.reward=reward # float to tune
        self.directions=[(1,0),(0,1),(-1,0),(0,-1)]
        self.iters = iters
        self.sunk_ships = []
        
        
    def update_board(self,board):
        self.size=len(board)
        self.eliminate_ships()
        self.guess_board = self.copy_board(board)

    def generate_heatmap(self):
        heatmap = list(np.zeros((self.size, self.size), dtype=int))
        for k in range(self.iters):
            dummy_board = self.copy_board(self.generate_dummy())
            for l in range(self.size):
                for m in range(self.size):
                    if dummy_board[l][m] == 'X' and self.guess_board[l][m] != 'O':
                        heatmap[l][m] += self.reward
        return heatmap

    def give_guess(self):
        hot_val = 0
        highest_list = []
        heatmap = self.generate_heatmap()
        for i in range(self.size):
            for j in range(self.size):
                if heatmap[i][j] > hot_val and self.guess_board[i][j] == '.':
                    hot_val, hot_pos, highest_list = heatmap[i][j], (i,j), [(i,j)]
                elif heatmap[i][j] == hot_val and self.guess_board[i][j]== '.':
                    highest_list.append((i,j))
        return random.choice(highest_list)

    def eliminate_ships(self):
        star_locs = self.find_val(self.copy_board(self.guess_board),'*')
        if star_locs:
            ship_size = 0
            for i in star_locs:
                if i not in self.sunk_ships:
                    self.sunk_ships.append(i)
                    ship_size += 1
            if ship_size > 0:
                ships.remove(ship_size)

    def generate_dummy(self):
        x_list = self.find_val(self.guess_board,'X')
        dot_list = self.find_val(self.guess_board,'.')
        dummy_board = self.copy_board(self.guess_board)
        for i in self.ships:
            if x_list:
                tile=random.choice(x_list)
                generated_ship = self.gen_ship_on_hit(tile,i,dummy_board)
                for j in generated_ship:
                    dummy_board[j[0]][j[1]]='X'
                x_list.remove(tile)
            else:
                tile=random.choice(dot_list)
                generated_ship = self.gen_ship_on_hit(tile,i,dummy_board)
                for j in generated_ship:
                    dummy_board[j[0]][j[1]]='X'
        return dummy_board
    
        
    def gen_ship_on_hit(self, tile, ship_size,dummy_board):
        directions=self.directions
        symbols = ['X','.'] #which tiles can a ship be generated on
        x_neighbor = self.rand_neighbor(tile,'X',directions,dummy_board) # gives a neighboring hit if one exists
        if x_neighbor: #try to find a ship on an X first
            ship_tiles = self.expand_ship(tile, x_neighbor[0],x_neighbor[1],ship_size,symbols,dummy_board)
            if len(ship_tiles) == ship_size:
                return ship_tiles
        #try perpendicular
        if x_neighbor:
            direction2 = self.perp_set(x_neighbor[1])
            x_neighbor2 = self.rand_neighbor(tile,'X',direction2,dummy_board)
            if x_neighbor2:
                ship_tiles = self.expand_ship(tile,x_neighbor2[0],x_neighbor2[1],ship_size,symbols,dummy_board)
                if len(ship_tiles) == ship_size:
                    return ship_tiles
        else:
            dot_neighbor = self.rand_neighbor(tile,'.',directions,dummy_board)
            if dot_neighbor:
                ship_tiles = self.expand_ship(tile,dot_neighbor[0],dot_neighbor[1],ship_size,symbols,dummy_board)
                if len(ship_tiles) == ship_size:
                    return ship_tiles
            if dot_neighbor:
                direction2 = self.perp_set(dot_neighbor[1])
                dot_neighbor2 = self.rand_neighbor(tile, '.',direction2,dummy_board)
                if dot_neighbor2:
                    ship_tiles = self.expand_ship(tile,dot_neighbor2[0],dot_neighbor2[1],ship_size,symbols,dummy_board)
                    if len(ship_tiles) == ship_size:
                        return ship_tiles
        return False

        
    def vect_add(self,v1,v2):
        return tuple(map(sum, zip(v1,v2)))
    
    def get_symbol(self,tile,board):
            return board[tile[0]][tile[1]]
    
    def rand_neighbor(self,tile,symbol,directions,board):
        #check neighbors in a random order to prevent bias
        random.shuffle(directions)
        for i in directions:
            neighbor = self.vect_add(i,tile)
            if self.valid_tile(neighbor,self.size):
                if self.get_symbol(tile,board) == symbol:
                    return neighbor, i
        #if none are valid, returns False
        return False
    
    def find_val(self,board,symbol): #return list of coordinates of the symbol
            n = len(board)
            hits = []
            for i in range(n):
                for j in range(n):
                    if board[i][j] == symbol:
                        hits.append((i,j))
            if hits:
                return hits
            return False
    
    def copy_board(self,board):
        new_board = []
        for i in board:
            new_row = []
            for j in i:
                new_row.append(j)
            new_board.append(new_row)
        return new_board
    
    def generate_blank(self,size):
        a='.'
        grid=[]
        for i in range(size):
            row=[]
            for j in range(size):
                row.append(a)
            grid.append(row)
        return grid
    
    def explore_dir(self,index_tile,direct,ship_tiles,goal_size,symbols,board):
            if len(ship_tiles) == goal_size:
                return ship_tiles
            while self.valid_tile(self.vect_add(index_tile,direct),self.size):
                new_tile = self.vect_add(index_tile,direct)
                if self.get_symbol(new_tile,board) not in symbols:
                    break
                ship_tiles.append(new_tile)
                if len(ship_tiles) == goal_size:
                    return ship_tiles
                index_tile = new_tile
            return ship_tiles
    
    def expand_ship(self,tile,neighbor,direct,ship_size,symbols,board):
            ship_tiles = [tile,neighbor]
            ship_tiles = self.explore_dir(neighbor,direct,ship_tiles,ship_size,symbols,board)
            opp_dir = (-1*direct[0],-1*direct[0])
            ship_tiles = self.explore_dir(tile,opp_dir,ship_tiles,ship_size,symbols,board)
            return ship_tiles
    
    def perp_set(self,direct):
            return [(direct[1],direct[0]),(-1*direct[1],-1*direct[0])]
    
    def valid_tile(self,tile,size):
            if 0 <= tile[0] < size:
                if 0 <= tile[1] < size:
                    return True
            return False
			
''' Weaker MDP agent using 5 iterations '''
def MDP_Agent_weak(grids, turn, ships):
    
    mdp_agent = Battleship_MDP(ships, 1, 5)
    
    def play_turn(grids, turn):
        mdp_agent.update_board(grids[turn])
        return mdp_agent.give_guess()

    return play_turn(grids, turn)
    
    
''' Stronger MDP agent using 15 iterations '''
def MDP_Agent_strong(grids, turn, ships):
    
    mdp_agent = Battleship_MDP(ships, 1, 15)
    
    def play_turn(grids, turn):
        mdp_agent.update_board(grids[turn])
        return mdp_agent.give_guess()

    return play_turn(grids, turn)

''' Probability Agent
    Determines most probable cells to shoot next based on the cells around it
'''
def Probablity_Agent(grids, turn, ships):
    
    def play_turn(grids, turn):
        clear = 0
        long_search = 0
        gridlists = grid_to_lists(grids[turn])
        #print(gridlists)
        #time.sleep(5)
        
        gridlists = calc_cell_values(gridlists)
        #print(gridlists)
        #time.sleep(5)
        
        target_grid = combine_gridlists(gridlists)
        #print(target_grid)
        #time.sleep(5)
        
        row, column = pick_target(target_grid)

        return row, column

    def calc_cell_values(gridlists):
        for i in range(len(gridlists)):
            cell_value = 0.1
            previous_blank = -1
            for j in range(len(gridlists[i])):
                if gridlists[i][j] == '*' or gridlists[i][j] == 'O':
                    gridlists[i][j] = 0.0
                    cell_value = 0.1
                    previous_blank = -1
                elif gridlists[i][j] == 'X':
                    gridlists[i][j] = 0.0
                    cell_value = cell_value + 0.1
                    if previous_blank >= 0:
                        gridlists[i][previous_blank] = cell_value
                else:
                    gridlists[i][j] = cell_value
                    previous_blank = j
        return gridlists

    def combine_gridlists(gridlists):
        target_grid = []
        midlength = int(len(gridlists)/2)
        for i in range(int(len(gridlists)/2)):
            target_row = gridlists[i]
            rowarray = []
            for j in range(len(gridlists[i])):
                rowvalue = gridlists[i][j] + gridlists[j+midlength][i] 
                rowarray.append(rowvalue)
            target_grid.append(rowarray)
        return target_grid

    def pick_target(target_grid):
        targets = []
        target_value = 0.0
        for i in range(len(target_grid)):
            for j in range(len(target_grid[i])):
                if target_grid[i][j] > target_value:
                    target_value = target_grid[i][j]
                    targets.clear()
                    targets.append((i,j))
                elif target_grid[i][j] == target_value:
                    targets.append((i,j))
        target = random_int(len(targets)-1)
        return targets[target]

    return play_turn(grids, turn)
    
''' Improved Probability Agent
    improves on previous probability agent by adding weight for adjacent blanks
'''
def Improved_Probablity_Agent(grids, turn, ships):
    
    def play_turn(grids, turn):
        clear = 0
        long_search = 0
        gridlists = grid_to_lists(grids[turn])
        #print(gridlists)
        
        gridlists = calc_cell_values(gridlists)
        #print(gridlists)
        
        target_grid = combine_gridlists(gridlists)
        #print(target_grid)
        
        row, column = pick_target(target_grid)
        #print(row, column)
        #time.sleep(1)
        return row, column

    def calc_cell_values(gridlists):
        for i in range(len(gridlists)):
            cell_value = 0.1
            previous_blank = -1
            adjacent_blanks = 0
            for j in range(len(gridlists[i])):
                if gridlists[i][j] == '*' or gridlists[i][j] == 'O':
                    gridlists[i][j] = 0.0
                    cell_value = 0.1
                    previous_blank = -1
                    adjacent_blanks = 0
                elif gridlists[i][j] == 'X':
                    gridlists[i][j] = 0.0
                    cell_value = cell_value + 1.0
                    if previous_blank >= 0:
                        gridlists[i][previous_blank] += cell_value
                else:
                    gridlists[i][j] = cell_value
                    previous_blank = j
                    adjacent_blanks += 1
                    cell_value = 0.1
                    
                if adjacent_blanks >= 2:
                    for k in range(adjacent_blanks):
                        gridlists[i][j-k] += 0.1
                        
        return gridlists

    def combine_gridlists(gridlists):
        target_grid = []
        midlength = int(len(gridlists)/2)
        for i in range(int(len(gridlists)/2)):
            target_row = gridlists[i]
            rowarray = []
            for j in range(len(gridlists[i])):
                rowvalue = gridlists[i][j] + gridlists[j+midlength][i] 
                rowarray.append(rowvalue)
            target_grid.append(rowarray)
        return target_grid

    def pick_target(target_grid):
        targets = []
        target_value = 0.0
        for i in range(len(target_grid)):
            for j in range(len(target_grid[i])):
                if target_grid[i][j] > target_value:
                    target_value = target_grid[i][j]
                    targets.clear()
                    targets.append((i,j))
                elif target_grid[i][j] == target_value:
                    targets.append((i,j))
        target = random_int(len(targets)-1)
        return targets[target]

    return play_turn(grids, turn)
    
''' Improved Seek Probability Agent
    Adds additional weight if seeking for larger blank spaces
'''
def Improved_Seek_Probablity_Agent(grids, turn, ships):
    
    def play_turn(grids, turn):
        clear = 0
        long_search = 0
        gridlists = grid_to_lists(grids[turn])
        #print(gridlists)
        
        gridlists = calc_cell_values(gridlists)
        #print(gridlists)
        
        target_grid = combine_gridlists(gridlists)
        #print(target_grid)
        
        row, column = pick_target(target_grid)
        #print(row, column)
        #time.sleep(1)
        return row, column

    def calc_cell_values(gridlists):
        for i in range(len(gridlists)):
            cell_value = 0.1
            previous_blank = -1
            adjacent_blanks = 0
            for j in range(len(gridlists[i])):
                if gridlists[i][j] == '*' or gridlists[i][j] == 'O':
                    gridlists[i][j] = 0.0
                    cell_value = 0.1
                    previous_blank = -1
                    adjacent_blanks = 0
                elif gridlists[i][j] == 'X':
                    gridlists[i][j] = 0.0
                    cell_value += 2.0
                    adjacent_blanks = 0
                    if previous_blank >= 0:
                        gridlists[i][previous_blank] += cell_value
                else:
                    gridlists[i][j] = cell_value
                    previous_blank = j
                    adjacent_blanks += 1
                    cell_value = 0.1
                    
                if adjacent_blanks >= 3:
                    for k in range(adjacent_blanks-1):
                        gridlists[i][j-k] = max(0.2*adjacent_blanks,gridlists[i][j-k])
                        
        return gridlists

    def combine_gridlists(gridlists):
        target_grid = []
        midlength = int(len(gridlists)/2)
        for i in range(int(len(gridlists)/2)):
            target_row = gridlists[i]
            rowarray = []
            for j in range(len(gridlists[i])):
                rowvalue = gridlists[i][j] + gridlists[j+midlength][i] 
                rowarray.append(rowvalue)
            target_grid.append(rowarray)
        return target_grid

    def pick_target(target_grid):
        targets = []
        target_value = 0.0
        for i in range(len(target_grid)):
            for j in range(len(target_grid[i])):
                if target_grid[i][j] > target_value:
                    target_value = target_grid[i][j]
                    targets.clear()
                    targets.append((i,j))
                elif target_grid[i][j] == target_value:
                    targets.append((i,j))
        target = random_int(len(targets)-1)
        return targets[target]

    return play_turn(grids, turn)
    
''' MCS Class '''
class Battleship_MCS:
    def __init__ (self,ships, board, reward=1, iters=10):
        # boards is a pair of n*n lists with 0 guess board, and 1 as true
        # on guess board, . is unknown, X is hit, * is sunk, O is miss
        # on hit board, . is no ship, otherwise ship
        # this agent checks for any hits on the board and prioritizes those, otherwise uses a heatmap to explore
        self.guess_board=[row[:] for row in board]
        self.size=len(board)
        self.ships=sorted([i for i in ships],reverse=True) # list of ship sizes
        self.reward=reward # unused
        self.directions=[(1,0),(0,1),(-1,0),(0,-1)]
        self.iters = iters # number of times to generate heatmap
        self.sunk_ships = []
        
    def update_board(self,board):
        self.eliminate_ships()
        self.guess_board = [row[:] for row in board] # get current board state

    def give_guess(self, heatmap):
        x_tiles = self.find_val('X') # get all 'X' tiles as starting points
        if not x_tiles: # if no 'X' tiles, go to the highest heatmap value
            return self.default_guess(heatmap)
        valid_guesses = self.get_valid_guesses_near_hits(x_tiles) # filter valid guesses around 'X' tiles
        max_val = max(heatmap[tile[0]][tile[1]] for tile in valid_guesses) # select the highest heatmap value from valid guesses
        highest_list = [tile for tile in valid_guesses if heatmap[tile[0]][tile[1]] == max_val] # still uses heatmap to select tile, but filters other hotspots
        return random.choice(highest_list) # choose a random tile from highest values

    def generate_heatmap(self):
        # generic function that runs other functions
        heatmap = np.zeros((self.size, self.size), dtype=int) #generate a blank
        for k in range(self.iters):
            dummy_board = self.generate_dummy() # each iteration a dummy board is generated from known information
            heatmap += np.array([[1 if cell == 'X' else 0 for cell in row] for row in dummy_board])
        return heatmap.tolist()

    def default_guess(self, heatmap):
        # fallback to the highest heatmap value when no hits exist
        max_val = max(map(max, heatmap))
        highest_list = [(i, j) for i in range(self.size) for j in range(self.size)
                        if self.guess_board[i][j] == '.' and heatmap[i][j] == max_val]
        return random.choice(highest_list)

    def get_valid_guesses_near_hits(self, x_tiles):
        # get all valid tiles near existing hits ('X') based on remaining ship sizes
        valid_tiles = set() #makes sure there are no dupes
        max_ship_size = max(self.ships) if self.ships else 0 #redundant check, keeps from breaking
        for x in x_tiles: #there are always x_tiles if this is called
            for direction in self.directions: #to check in all directions
                for step in range(1, max_ship_size + 1):
                    neighbor = (x[0] + step * direction[0], x[1] + step * direction[1])
                    if self.valid_tile(neighbor) and self.guess_board[neighbor[0]][neighbor[1]] == '.':
                        valid_tiles.add(neighbor)
        return list(valid_tiles)

    def eliminate_ships(self):
        # called every turn so we dont look for ships that have already been sunk
        star_locs = self.find_val('*') # all tiles with a star
        if star_locs:
            new_sunk = [loc for loc in star_locs if loc not in self.sunk_ships] # see if any new sunk ships added
            self.sunk_ships.extend(new_sunk)
            ship_size = len(new_sunk)
            if ship_size in self.ships:
                self.ships.remove(ship_size) # stop looking for ship if it has been sunk

    def generate_dummy(self):
        x_list = self.find_val('X') # get list of pairs (x,y) of places labeled X
        if x_list:
            random.shuffle(x_list)
        dot_list = self.find_val('.') # get list of pairs (x,y) of placed labeled
        if dot_list:
            random.shuffle(dot_list)
        dummy_board = [row[:]for row in self.guess_board] # get a copy of the current guess board
        ships_to_place = self.ships[:] # get a copy of the ships that need to be placed
        previous_ships_count = len(ships_to_place)
        self.place_ships_on_hits(dummy_board,x_list,ships_to_place) # place all hits first
        self.place_ships(dummy_board,dot_list,ships_to_place,'.') # if there are hits left
        if len(ships_to_place) == previous_ships_count:
            #print("No progress made in ship placement") # if something breaks, gives error
            return dummy_board
        return [['X' if tile == 'G' else tile for tile in row] for row in dummy_board]

    def place_ships(self,board,tile_list,ships_to_place,symbol):
        # input: current dummy board state, list of tiles to place ships on, list of ships that need placed, the symbol to put them on
        # output: changes tiles on the input board
        for ship_size in ships_to_place[:]:
            for tile in tile_list:
                gen_ship = self.gen_ship_on_hit(tile,ship_size,board,symbol)
                if gen_ship:
                    for tile1 in gen_ship:
                        board[tile1[0]][tile1[1]] = 'G'
                    ships_to_place.remove(ship_size)
                    break
                    
    def place_ships_on_hits(self, board, x_list, ships_to_place):
        while x_list and ships_to_place:
            progress_made = False  # track if progress is made during this iteration
            for ship_size in ships_to_place[:]:  # iter over ships to place
                valid_ship_found = False  # track if a valid ship is placed for this size
                for tile in x_list[:]:  # iterate over available 'X' tiles
                    generated_ship = self.gen_ship_on_multiple_hits(tile, ship_size, board)
                    if generated_ship:  # ship placement succeeded
                        # mark the generated ship on the board
                        for ship_tile in generated_ship:
                            board[ship_tile[0]][ship_tile[1]] = 'G'
                            if ship_tile in x_list:  # remove used tiles
                                x_list.remove(ship_tile)
                        ships_to_place.remove(ship_size)  # remove placed ship
                        progress_made = True
                        valid_ship_found = True
                        break  # break to place the next ship size
                if not valid_ship_found:
                    ships_to_place.remove(ship_size)
            # Exit if no progress was made in this iter
            if not progress_made:
                #print("No progress made in placing ships on hits. Breaking loop.")
                break

    def gen_ship_on_hit(self, tile, ship_size, board, symbol):
        # used instead of multiple hits, actually generates ship on an unchecked tile
        for direction in self.directions:
            ship_tiles = self.expand_ship(tile, direction, ship_size, board, symbol)
            if len(ship_tiles) == ship_size:
                return ship_tiles
        return None
            
    def print_grid(self, grid: list): # used in debugging
        row_labels = 'A B C D E F G H I J K L M N'.split()
        print ('   ' + ' '.join([f'{i}' for i in range(len(grid))]))
        
        for i, row in enumerate(grid):
            print(f'{row_labels[i]:2}', end=' ')
            for cell in row:
                print(cell, end=' ')
            print()
        
    def gen_ship_on_multiple_hits(self, tile, ship_size, board):
        # used when we need to try to place a ship across multiple hits
        for direction in self.directions:
            ship_tiles = [tile]
            for step in range(1, ship_size):
                neighbor = (tile[0] + step * direction[0], tile[1] + step * direction[1])
                if not self.valid_tile(neighbor) or board[neighbor[0]][neighbor[1]] not in {'X', '.'}:
                    break # if it finds a miss or a sunk, stop looking in that direction
                ship_tiles.append(neighbor)
            if len(ship_tiles) == ship_size:
                return ship_tiles  # valid ship found
        return False
    
    def find_val(self,symbol): # return list of coordinates of the symbol
            return [(i, j) for i in range(self.size) for j in range(self.size) if self.guess_board[i][j] == symbol]
 
    def expand_ship(self, tile, direction, ship_size, board, symbol):
        ship_tiles = []
        for step in range(-ship_size + 1, ship_size): # get the complete range of possible placements on the tile
            neighbor = (tile[0] + step * direction[0], tile[1] + step * direction[1]) # get tuple of neighbor
            if self.valid_tile(neighbor) and board[neighbor[0]][neighbor[1]] in {symbol, '.'}: # every step must be empty, or a hit
                ship_tiles.append(neighbor) # adds valid tile to the ship being generated
                if len(ship_tiles) == ship_size:
                    return ship_tiles
        return []
    
    def valid_tile(self, tile):
        return 0 <= tile[0] < self.size and 0 <= tile[1] < self.size
		
def MCS_Agent_X1(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 1)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X2(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 2)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X3(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 3)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X4(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 4)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)
    
def MCS_Agent_X5(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 5)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X10(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 10)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X20(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 20)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X100(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 100)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X200(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 200)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X1000(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 1000)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)

def MCS_Agent_X10000(grids, turn, ships):
    
    mcs_agent = Battleship_MCS(ships, grids[turn], 1, 10000)
    
    def play_turn(grids, turn):
        mcs_agent.update_board(grids[turn])
        heatmap = mcs_agent.generate_heatmap()
        return mcs_agent.give_guess(heatmap)

    return play_turn(grids, turn)
	
class BAgent:
    def __init__(self, agent, name, average=0, best=100, worst=0, avg_first_sunk=100):
        self.agent = agent
        self.name = name
        self.average = average
        self.best = best
        self.worst = worst
        self.avg_first_sunk = avg_first_sunk
        
    def __repr__(self):
        return repr((self.agent, self.name, self.average, self.best, self.worst, self.avg_first_sunk))
        
def agent_trials(agents, grid_size, ships, trial_games, printgrids=False):
    for agent in agents:
        wins = np.array([0,0,0,0])
        games = trial_games
        best = 100
        worst = 0
        while games > 0:
            if printgrids == True:
                time.sleep(1)
            elif games % 10 == 0:
                clear_output(wait=True)
                print(str(games) + " trials remaining: " + agent.name)
                
            result = play(grid_size, ships, agents=[agent.agent, agent.agent], \
                         hidden=[False, False], title=str(games) + " trials remaining: " + agent.name,
                         printgrids=printgrids)
            wins += result
            if result[2] < best:
                best = result[2]
            if result[2] > worst:
                worst = result[2]
                
            games = games - 1
            
                
        agent.average = int(wins[2]/trial_games)
        agent.best = best
        agent.worst = worst
        agent.avg_first_sunk = int(wins[3]/trial_games)

    return sorted(agents, key=lambda agent: agent.average)

def agent_reset(agents):
    for agent in agents:
        agent.average = 0
    return sorted(agents, key=lambda agent: agent.average)
	
grid_size = 10
ships = [5, 4, 3, 3, 2]


agents = [
    BAgent(MCS_Agent_X20, "MCS_Agent_X20", 0),
    BAgent(Random_Agent, "Random_Agent", 0),
    BAgent(Improved_Random_Agent, "Improved_Random_Agent", 0),
    BAgent(Probablity_Agent, "Probablity_Agent", 0),
    BAgent(Improved_Probablity_Agent, "Improved_Probablity_Agent", 0),
    BAgent(Improved_Seek_Probablity_Agent, "Improved_Seek_Probablity_Agent", 0),
    BAgent(MDP_Agent_weak, "MDP_Agent_weak", 0),
    BAgent(MDP_Agent_strong, "MDP_Agent_strong", 0),
    BAgent(MCS_Agent_X1, "MCS_Agent_X1", 0),
    BAgent(MCS_Agent_X2, "MCS_Agent_X2", 0),
    BAgent(MCS_Agent_X3, "MCS_Agent_X3", 0),
    BAgent(MCS_Agent_X4, "MCS_Agent_X4", 0),
    BAgent(MCS_Agent_X5, "MCS_Agent_X5", 0),
    BAgent(MCS_Agent_X10, "MCS_Agent_X10", 0),
    BAgent(MCS_Agent_X100, "MCS_Agent_X100", 0),
    BAgent(MCS_Agent_X200, "MCS_Agent_X200", 0),
    BAgent(MCS_Agent_X1000, "MCS_Agent_X1000", 0),
    BAgent(MCS_Agent_X10000, "MCS_Agent_X10000", 0)
]

valid = False
while not valid:
    trials_value = input("Enter number of Trial games used to rank agents: (0-100)")
    try:
        trials = int(trials_value)
        if trials >= 0 and trials <= 999999:
            valid = True
    except:
        print("Invalid number of trials.")

if trials > 0: 
    sorted_agents = agent_trials(copy.deepcopy(agents), grid_size, ships, trials, printgrids=False)
    chart_results(sorted_agents, "Average shots to win after " + str(trials) + " games:")
    record_results(sorted_agents,trials,"BattleshipAdmiralsResults.csv",'a')

''' Play Human vs Best AI Agent, or play Best 2 AI Agents vs each other '''
valid = False
while not valid:
    player_value = input("Play Human against AI? (Y or N)")
    try:
        player = str(player_value).upper()[:1]
        if player == 'Y' or player == 'N':
            valid = True
    except:
        print("Invalid Answer.")
        
valid = False
while not valid:
    games_value = input("Enter number of Games to Play: (0-100)")
    try:
        play_games = int(games_value)
        if play_games >= 0 and play_games <= 100:
            valid = True
    except:
        print("Invalid number of games.")

if player == 'Y':
    player1=BAgent(Human_Player, "Human_Player", 0)
    player1_hidden = True
else:
    player1=sorted_agents[1]
    player1_hidden = False
player2=sorted_agents[0]

wins = np.array([0,0,0,0])
games = play_games
while games > 0:
    wins += play(grid_size, ships, agents=[player1.agent, player2.agent], hidden=[player1_hidden, False], title=player1.name + " vs " + player2.name)
    games = games - 1
    time.sleep(2)

if wins[0] > wins[1]:
    print(player1.name + " wins!!! " + str(wins[0]) + " to " + str(wins[1]))
elif wins[1] > wins[0]:
    print(player2.name + " wins!!! " + str(wins[1]) + " to " + str(wins[0]))
else:
    print("TIED!!! " + str(wins[0]) + " to " + str(wins[1]))

print("Average Shots to Win: " + str(int(wins[2]/play_games)))
