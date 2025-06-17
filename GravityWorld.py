import random 

# --- GravityWorld Class ---
class Tile:
    EMPTY = 0
    BLOCK = 1
    AGENT = 2
    FOOD = 3

# --- GravityWorld Class ---
class Tile:
    EMPTY = 0
    BLOCK = 1
    AGENT = 2
    FOOD = 3


class GravityWorld:
    def __init__(self, width=10, height=10): # num_random_blocks removed, terrain is procedural
        self.width = width
        self.height = height
        self.grid = [[Tile.EMPTY for _ in range(width)] for _ in range(height)]

        self.agent_pos = (0,0)
        self.food_pos = (0,0)
        self.agent_holding = False

        # Initialize solid ground layer at the bottom
        for x in range(self.width):
            self.place_tile(x, self.height - 1, Tile.BLOCK)

        self.generate_step_terrain() # Call the new procedural terrain generation

    def generate_step_terrain(self):
        # The y-coordinate of the ground is self.height - 1.
        # Max possible column height (blocks above ground layer, excluding the ground itself)
        # User wants height 0-25 for world height 50. So, max is roughly half world height.
        MAX_COLUMN_HEIGHT_ABOVE_GROUND = self.height // 2 - 1
        if MAX_COLUMN_HEIGHT_ABOVE_GROUND < 0:
            MAX_COLUMN_HEIGHT_ABOVE_GROUND = 0

        # Initialize the height of the first column
        current_column_height = random.randint(0, MAX_COLUMN_HEIGHT_ABOVE_GROUND)

        for x in range(self.width):
            # Place blocks in the current column from the ground up to `current_column_height`
            for y_offset in range(current_column_height):
                block_y = self.height - 2 - y_offset # Start from one above ground, going upwards
                if block_y >= 0: # Ensure block_y is within grid bounds
                     self.place_tile(x, block_y, Tile.BLOCK)

            # Determine height for the next column (only if not the last column)
            if x < self.width - 1:
                height_change = random.choice([-1, 0, 1])
                next_column_height = current_column_height + height_change
                
                # Ensure next_column_height stays within valid bounds (0 to MAX_COLUMN_HEIGHT_ABOVE_GROUND)
                next_column_height = max(0, min(MAX_COLUMN_HEIGHT_ABOVE_GROUND, next_column_height))
                
                current_column_height = next_column_height

    def place_tile(self, x, y, kind):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = kind

    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def apply_gravity(self, agent_positions, food_positions): # Modified to accept lists of positions
        # Store initial positions for return
        original_agent_positions = list(agent_positions)
        original_food_positions = list(food_positions)
        
        # Apply gravity to falling blocks
        for y in reversed(range(self.height - 1)):
            for x in range(self.width):
                if self.grid[y][x] == Tile.BLOCK and self.grid[y + 1][x] == Tile.EMPTY and (y + 1 < self.height -1):
                    self.grid[y][x] = Tile.EMPTY
                    self.grid[y + 1][x] = Tile.BLOCK
                    # If any entity (food or agent) was on this falling block, they should fall with it.
                    # This is handled by entity-specific gravity loops below, or if they just 'land' on it.

        # Apply gravity to Food
        # Create a list of food positions that are currently on the grid to avoid modifying while iterating
        current_food_on_grid = []
        for y_grid in range(self.height):
            for x_grid in range(self.width):
                if self.grid[y_grid][x_grid] == Tile.FOOD:
                    current_food_on_grid.append((x_grid, y_grid))

        updated_food_positions = []
        for x, y in current_food_on_grid:
            if y < self.height - 1 and self.grid[y + 1][x] == Tile.EMPTY:
                self.grid[y][x] = Tile.EMPTY # Clear old spot
                self.grid[y + 1][x] = Tile.FOOD # Place in new spot
                updated_food_positions.append((x, y + 1))
            else:
                updated_food_positions.append((x, y)) # No fall, retain original position
        
        # Apply gravity to Agents
        # Similar to food, collect agent positions then apply gravity
        current_agents_on_grid = []
        for y_grid in range(self.height):
            for x_grid in range(self.width):
                if self.grid[y_grid][x_grid] == Tile.AGENT:
                    current_agents_on_grid.append((x_grid, y_grid))

        updated_agent_positions = []
        for x, y in current_agents_on_grid:
            if y < self.height - 1 and self.grid[y + 1][x] in [Tile.EMPTY, Tile.FOOD]:
                self.grid[y][x] = Tile.EMPTY # Clear old spot
                self.grid[y + 1][x] = Tile.AGENT # Place in new spot
                updated_agent_positions.append((x, y + 1))
            else:
                updated_agent_positions.append((x, y)) # No fall, retain original position

        return updated_agent_positions, updated_food_positions # Return updated positions



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
        return self.agent_pos == self.food_pos or self.food_pos == (None, None)

    def render(self):
        render_grid = [row[:] for row in self.grid]

        if self.food_pos != (None, None) and self.get_tile(*self.food_pos) == Tile.FOOD:
            render_grid[self.food_pos[1]][self.food_pos[0]] = Tile.FOOD

        ax, ay = self.agent_pos
        if self.get_tile(ax, ay) == Tile.AGENT:
             render_grid[ay][ax] = Tile.AGENT

        if self.agent_holding:
            if ay > 0:
                render_grid[ay-1][ax] = Tile.BLOCK

        return render_grid

    def column_surface_height(self, x):
        """Return the y-coordinate of the topmost block in column ``x``."""
        for y in range(self.height - 2, -1, -1):
            if self.grid[y][x] == Tile.BLOCK:
                return y
        return self.height - 1

    def enforce_traversable(self):
        for x in range(self.width - 1):
            h_cur = self.column_surface_height(x)
            h_next = self.column_surface_height(x + 1)

            while h_cur < h_next - 1:
                if h_cur > 0:
                    self.place_tile(x, h_cur - 1, Tile.BLOCK)
                    h_cur -= 1
                else:
                    break

            while h_next < h_cur - 1:
                if h_next > 0:
                    self.place_tile(x + 1, h_next - 1, Tile.BLOCK)
                    h_next -= 1
                else:
                    break
