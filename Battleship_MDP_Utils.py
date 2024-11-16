import random

def vect_add(v1,v2):
    return tuple(map(sum, zip(v1,v2)))

def get_symbol(tile,board):
        return board[tile[0]][tile[1]]

def rand_neighbor(tile,symbol,directions,size,board):
    #check neighbors in a random order to prevent bias
    random.shuffle(directions)
    for i in directions:
        neighbor = vect_add(i,tile)
        if valid_tile(neighbor,size):
            if get_symbol(tile,board) == symbol:
                return neighbor, i
    #if none are valid, returns False
    return False

def find_val(board,symbol): #return list of coordinates of the symbol
        n = len(board)
        hits = []
        for i in range(n):
            for j in range(n):
                if board[i][j] == symbol:
                    hits.append((i,j))
        if hits:
            return hits
        return False

def generate_blank(size):
    a='.'
    grid=[]
    for i in range(size):
        row=[]
        for j in range(size):
            row.append(a)
        grid.append(row)
    return grid

def explore_dir(index_tile,direct,ship_tiles,goal_size,symbols,board):
        size = len(board)
        if len(ship_tiles) == goal_size:
            return ship_tiles
        while valid_tile(vect_add(index_tile,direct),size):
            new_tile = vect_add(index_tile,direct)
            if get_symbol(new_tile,board) not in symbols:
                break
            ship_tiles.append(new_tile)
            if len(ship_tiles) == goal_size:
                return ship_tiles
            index_tile = new_tile
        return ship_tiles

def expand_ship(tile,neighbor,direct,ship_size,symbols,board):
        ship_tiles = [tile,neighbor]
        ship_tiles = explore_dir(neighbor,direct,ship_tiles,ship_size,symbols,board)
        opp_dir = (-1*direct[0],-1*direct[0])
        ship_tiles = explore_dir(tile,opp_dir,ship_tiles,ship_size,symbols,board)
        return ship_tiles

def perp_set(direct):
        return [(direct[1],direct[0]),(-1*direct[1],-1*direct[0])]

def valid_tile(tile,size):
        if 0 <= tile[0] < size:
            if 0 <= tile[1] < size:
                return True
        return False

