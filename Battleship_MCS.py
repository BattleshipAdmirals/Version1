import random
import numpy as np
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
            print("No progress made in ship placement") # if something breaks, gives error
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
                print("No progress made in placing ships on hits. Breaking loop.")
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
