import random
import numpy as np
import Battleship_MDP_Utils as utl
class Battleship_MDP:
    def __init__ (self,ships, reward):
        # boards is a pair of n*n lists with 0 guess board, and 1 as true
        # on guess board, . is unknown, X is hit, * is sunk, 0 is miss
        # on hit board, . is no ship, otherwise ship
        self.guess_board=[]
        self.size=0
        self.ships=ships # list of ship sizes
        self.reward=reward # float to tune
        self.directions=[(1,0),(0,1),(-1,0),(0,-1)]
        self.iters = 10
        
        
    def update_board(self,board):
        self.clear()
        new_board = []
        for i in board:
            new_row = []
            for j in i:
                new_row.append(j)
            new_board.append(new_row)
        self.guess_board=new_board
        self.size=len(board)
        return new_board

    def clear(self):
        blank = utl.generate_blank(self.size)
        self.dummy_board=blank

    def generate_heatmap(self):
        heatmap = list(np.zeros((self.size, self.size), dtype=int))
        for k in range(self.iters):
            dummy_board = self.generate_dummy()
            for l in range(self.size):
                for m in range(self.size):
                    if dummy_board[l][m] == 'X':
                        heatmap[l][m] += self.reward
        return heatmap

    def give_guess(self,heatmap):
        x_list = utl.find_val(self.guess_board,'X')
        if x_list == False:
            x_list = []
        hot_val = 0
        highest_list = []
        for i in range(self.size):
            for j in range(self.size):
                if heatmap[i][j] > hot_val and (i,j) not in x_list:
                    hot_val = heatmap[i][j]
                    hot_pos = (i,j)
                    highest_list =[(i,j)]
                elif heatmap[i][j] == hot_val and (i,j) not in x_list:
                    highest_list.append((i,j))
        return random.choice(highest_list)

    def eliminate_ships(self):
        ships_to_find = self.ships
        directions=self.directions
        symbol=['*']
        check_board = self.guess_board
        for i in check_board:
            for j in i:
                if j == '*':
                    tile = (check_board[i],j)
                    neighbor = rand_neighbor(tile,'*',directions,self.size,self.guess_board)
                    if neighbor:
                        for k in self.ships:
                            ship_tiles = utl.expand_ship(tile,neighbor[0],neighbor[1],k,symbol,self.guess_board)
                            if len(ship_tiles) == k:
                                ships_to_find.pop(k)
                                for l in ship_tiles:
                                    check_board[l[0]][l[1]] = '0'
                                break                 
        return ships_to_find


    def generate_dummy(self):
        ships_left=self.eliminate_ships()
        x_list = utl.find_val(self.guess_board,'X')
        dot_list = utl.find_val(self.guess_board,'.')
        dummy_board = utl.generate_blank(self.size)
        for i in ships_left:
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
        symbols = ['X','.']
        x_neighbor = utl.rand_neighbor(tile,'X',directions,self.size,dummy_board)
        #try to find a ship on an x first
        if x_neighbor:
            ship_tiles = utl.expand_ship(tile, x_neighbor[0],x_neighbor[1],ship_size,symbols,dummy_board)
            if len(ship_tiles) == ship_size:
                return ship_tiles
        #try perpendicular
        if x_neighbor:
            direction2 = utl.perp_set(x_neighbor[1])
            x_neighbor2 = utl.rand_neighbor(tile,'X',direction2,self.size,dummy_board)
            if x_neighbor2:
                ship_tiles = utl.expand_ship(tile,x_neighbor2[0],x_neighbor2[1],ship_size,symbols,dummy_board)
                if len(ship_tiles) == ship_size:
                    return ship_tiles
        else:
            dot_neighbor = utl.rand_neighbor(tile,'.',directions,self.size,dummy_board)
            if dot_neighbor:
                ship_tiles = utl.expand_ship(tile,dot_neighbor[0],dot_neighbor[1],ship_size,symbols,dummy_board)
                if len(ship_tiles) == ship_size:
                    return ship_tiles
            if dot_neighbor:
                direction2 = utl.perp_set(dot_neighbor[1])
                dot_neighbor2 = utl.rand_neighbor(tile, '.',direction2,self.size,dummy_board)
                if dot_neighbor2:
                    ship_tiles = utl.expand_ship(tile,dot_neighbor2[0],dot_neighbor2[1],ship_size,symbols,dummy_board)
                    if len(ship_tiles) == ship_size:
                        return ship_tiles
        return False
