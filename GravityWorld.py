# --- GravityWorld Class ---
class Tile:
    EMPTY = 0
    BLOCK = 1
    AGENT = 2
    FOOD = 3


class GravityWorld:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = [[Tile.EMPTY for _ in range(width)] for _ in range(height)]

        self.agent_pos = (0,0) # Agent position will be set by external functions
        self.food_pos = (0,0)  # Food position will be set by external functions
        self.agent_holding = False

        # Initialize solid ground layer at the bottom
        for x in range(self.width):
            self.place_tile(x, self.height - 1, Tile.BLOCK)

        # Agent is NOT placed here anymore. Setup is handled by train/test functions.

    def place_tile(self, x, y, kind):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = kind

    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def apply_gravity(self):
        for y in reversed(range(self.height - 1)):
            for x in range(self.width):
                if self.grid[y][x] == Tile.BLOCK and self.grid[y + 1][x] == Tile.EMPTY and (y + 1 < self.height -1):
                    self.grid[y][x] = Tile.EMPTY
                    self.grid[y + 1][x] = Tile.BLOCK
                    if (x, y) == self.food_pos:
                        self.food_pos = (x, y + 1)

        ax, ay = self.agent_pos
        if ay < self.height - 1 and self.grid[ay + 1][ax] == Tile.EMPTY:
            self.grid[ay][ax] = Tile.EMPTY
            self.grid[ay + 1][ax] = Tile.AGENT
            self.agent_pos = (ax, ay + 1)


    def move_agent(self, dx):
        x, y = self.agent_pos
        new_x = x + dx
        if 0 <= new_x < self.width and self.grid[y][new_x] in [Tile.EMPTY, Tile.FOOD] \
           and (y == self.height - 1 or self.get_tile(x, y + 1) == Tile.BLOCK):

            # Before moving, clear the old agent spot.
            self.place_tile(x, y, Tile.EMPTY)

            # If moving onto food, consider it collected by moving onto it
            if self.grid[y][new_x] == Tile.FOOD:
                self.food_pos = (None, None) # Food is "gone" from the world grid

            self.agent_pos = (new_x, y) # Update agent position
            self.place_tile(new_x, y, Tile.AGENT) # Place agent at new spot
            return True
        return False

    def jump(self, dx):
        x, y = self.agent_pos
        new_x = x + dx
        on_ground_or_block = (y == self.height - 1 or self.grid[y+1][x] == Tile.BLOCK)

        if on_ground_or_block and y > 0 and 0 <= new_x < self.width and self.grid[y - 1][new_x] == Tile.EMPTY:
            self.place_tile(x, y, Tile.EMPTY)
            self.agent_pos = (new_x, y - 1)
            self.place_tile(new_x, y - 1, Tile.AGENT)
            return True
        return False

    def pick_up(self):
        if self.agent_holding:
            return False
        x, y = self.agent_pos
        if y + 1 < self.height and self.grid[y + 1][x] == Tile.BLOCK:
            self.grid[y + 1][x] = Tile.EMPTY
            self.agent_holding = True
            return True
        return False

    def place_block(self):
        if not self.agent_holding:
            return False
        x, y = self.agent_pos
        if y + 1 < self.height and self.grid[y + 1][x] == Tile.EMPTY and (y + 1 < self.height -1):
            self.grid[y + 1][x] = Tile.BLOCK
            self.agent_holding = False
            return True
        return False

    def is_food_collected(self):
        # Food is collected if agent is at food's position.
        # Note: food_pos is set to (None,None) in move_agent when collected
        return self.agent_pos == self.food_pos or self.food_pos == (None, None)

    def render(self):
        # Create a deep copy of the grid for rendering to avoid modifying the actual game state
        render_grid = [row[:] for row in self.grid]

        # Draw Food (if still present) first, so Agent can draw over it
        if self.food_pos != (None, None) and self.get_tile(*self.food_pos) == Tile.FOOD:
            render_grid[self.food_pos[1]][self.food_pos[0]] = Tile.FOOD

        # Draw Agent on top of everything else
        ax, ay = self.agent_pos
        if self.get_tile(ax, ay) == Tile.AGENT:
             render_grid[ay][ax] = Tile.AGENT

        # If agent is holding a block, render it above the agent for visualization purposes
        if self.agent_holding:
            if ay > 0:
                render_grid[ay-1][ax] = Tile.BLOCK

        return render_grid