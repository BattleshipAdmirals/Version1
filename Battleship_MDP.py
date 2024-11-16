import random
import numpy as np
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
