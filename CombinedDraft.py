import math
import random
from IPython.display import clear_output
import time
import numpy as np
import Battleship_MDP as mdp
    

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

def validate_size(size: int):
    side_length = int(math.sqrt(size))
    if side_length * side_length != size:
        raise ValueError("Size must create perfect square")

def print_game(grids, shots, hits, hidden):
    print("                Battleship Admirals")
    print()
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

def play(grid_size, ships, agents, hidden=(False, False)):
    grid1, grid2, grid3, grid4, dict1, dict2= setup_game(grid_size, ships)
    grids = [grid1,grid2,grid3,grid4]
    sunk = [dict1,dict2]

    hits_to_win = sum(ships)
    max_shots = grid_size*grid_size
    shots = [0, 0]
    hits = [0, 0]
    turn = 0
    print_game(grids, shots, hits, hidden)
    
    while shots[0] < max_shots and shots[1] < max_shots and hits[0] < hits_to_win and hits[1] < hits_to_win:
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
        shots[turn] = shots[turn] + 1
        if turn == 0:
            turn = 1
        else:
            turn = 0

        clear_output(wait=True)
        print_game(grids, shots, hits, hidden)
        if hits[0] >= hits_to_win:
            print('Player 1 Wins!!!')
            return np.array([1,0])
        if hits[1] >= hits_to_win:
            print('Player 2 Wins!!!')
            return np.array([0,1])


#Improved Random Agent?
#If next to miss and not next to hit, skip
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
                print('LONG SEARCH', turn, row, column)
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

def MDP_Agent_strong(grids, turn, ships):
    
    mdp_agent = mdp.Battleship_MDP(ships, 1, 15)
    
    def play_turn(grids, turn):
        mdp_agent.update_board(grids[turn])
        return mdp_agent.give_guess()

    return play_turn(grids, turn)

def MDP_Agent_weak(grids, turn, ships):
    
    mdp_agent = mdp.Battleship_MDP(ships, 1, 5)
    
    def play_turn(grids, turn):
        mdp_agent.update_board(grids[turn])
        return mdp_agent.give_guess()

    return play_turn(grids, turn)
    
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
    
grid_size = 10
ships = [5, 4, 3, 3, 2]
games = 100

agent1=Random_Agent
agent2=Improved_Random_Agent
agent3=Probablity_Agent
agent4=Improved_Probablity_Agent
agent5=Improved_Seek_Probablity_Agent

agent6=MDP_Agent_strong
agent7=MDP_Agent_weak
agent8=Human_Player

wins = np.array([0,0])
while games > 0:
    wins += play(grid_size, ships, agents=[agent8, agent4], hidden=[True, False])
    games = games - 1
    #time.sleep(5)
    
print(wins)
