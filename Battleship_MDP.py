import random
import Battleship_MDP_Utils as utl
class Battleship_MDP:
    def __init__ (self, boards, ships, reward):
        # boards is a pair of n*n lists with 0 guess board, and 1 as true
        # on guess board, . is unknown, X is hit, * is sunk, 0 is miss
        # on hit board, . is no ship, otherwise ship
        self.guess_board=boards[0]
        self.true_board=boards[1]
        self.size=len(boards[0])
        self.ships=ships # list of ship sizes
        self.reward=reward # float to tune
        self.directions=[(1,0),(0,1),(-1,0),(0,-1)]

    def update_board(self,board):
        self.guess_board=board

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

    def find_hit(self):
        check_board = self.guess_board
        n = self.size
        for i in range(n):
            for j in range(n):
                if check_board[i][j] == 'X':
                    return (i,j)
        return False
        
    def gen_ship_on_hit(self, tile, ship_size):
        directions=self.directions
        symbols = ['X','.']
        x_neighbor = utl.rand_neighbor(tile,'X',directions,self.size,self.guess_board)
        #try to find a ship on an x first
        if x_neighbor:
            ship_tiles = utl.expand_ship(tile, x_neighbor[0],x_neighbor[1],ship_size,symbols,self.guess_board)
            if len(ship_tiles) == ship_size:
                return ship_tiles
        #try perpendicular
        if x_neighbor:
            direction2 = utl.perp_set(x_neighbor[1])
            x_neighbor2 = utl.rand_neighbor(tile,'X',direction2,self.size,self.guess_board)
            if x_neighbor2:
                ship_tiles = utl.expand_ship(tile,x_neighbor2[0],x_neighbor2[1],ship_size,symbols,self.guess_board)
                if len(ship_tiles) == ship_size:
                    return ship_tiles
        else:
            dot_neighbor = utl.rand_neighbor(tile,'.',directions,self.size,self.guess_board)
            if dot_neighbor:
                ship_tiles = utl.expand_ship(tile,dot_neighbor[0],dot_neighbor[1],ship_size,symbols,self.guess_board)
                if len(ship_tiles) == ship_size:
                    return ship_tiles
            if dot_neighbor:
                direction2 = utl.perp_set(dot_neighbor[1])
                dot_neighbor2 = utl.rand_neighbor(tile, '.',direction2,self.size)
                if dot_neighbor2:
                    ship_tiles = utl.expand_ship(tile,dot_neighbor2[0],dot_neighbor2[1],ship_size,symbols,self.guess_board)
                    if len(ship_tiles) == ship_size:
                        return ship_tiles
        return False
