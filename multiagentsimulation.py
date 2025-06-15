import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import colors, cmap, bounds, norm # Import color config

# Import classes from your existing modules
from Agent import Agent
from GravityWorld import GravityWorld, Tile
from mutations import mutate_agent # Assuming you want to use your mutation logic for reproduction

# --- Configuration Parameters ---
SAVE_DIRECTORY = "trained_agents"
WORLD_WIDTH = 50
WORLD_HEIGHT = 50
INITIAL_AGENT_ENERGY = 100.0
ENERGY_DEPLETION_RATE = 1 # Energy lost per simulation step
FOOD_ENERGY_RECHARGE = 50.0 # Energy gained from eating food
FOOD_SPAWN_INTERVAL = 25 # Every N steps, try to spawn food
MAX_FOOD_ON_MAP = 5 # Maximum number of food items allowed at once
REPRODUCTION_ENERGY_THRESHOLD = 101.0 # Energy needed to reproduce
REPRODUCTION_COST = 10.0 # Energy cost of reproduction
MUTATION_RATE = 0.1 # Rate for mutations during reproduction
MUTATION_STRENGTH = 0.5 # Strength for mutations during reproduction
NEURON_COUNT = 50 # Consistent neuron count for agents
INPUT_NOISE_STD = .1

NUM_SIMULATION_STEPS = 1_000_000 # Total steps for the multi-agent simulation


# --- Helper Functions ---
def load_population(directory, num=None):
    """Loads a population of Agent objects from a specified directory.
    Prioritizes loading from 'sim_population.pkl' if it exists,
    otherwise loads individual agent files.
    
    Args:
        directory (str): The path to the directory containing agent pickle files.
        num (int, optional): The desired number of agents to load. If more agents
                             are available, a random subset will be loaded.
                             If None, all available agents will be loaded.
    Returns:
        list: A list of loaded Agent objects.
    """
    population = []
    filepath_sim_population = os.path.join(directory, "sim_population.pkl")

    # Strategy 1: Try to load from a single 'sim_population.pkl' file
    if os.path.exists(filepath_sim_population):
        try:
            with open(filepath_sim_population, 'rb') as f:
                loaded_all_agents = pickle.load(f)
            print(f"Loaded population from '{filepath_sim_population}'.")
            
            # Ensure agents have energy attribute if loaded from old saves
            # (Assuming INITIAL_AGENT_ENERGY is accessible in this scope)
            for agent in loaded_all_agents:
                if not hasattr(agent, 'energy'):
                    agent.energy = INITIAL_AGENT_ENERGY # Default energy for compatibility
            
            # Apply num filtering if requested
            if num is not None and len(loaded_all_agents) > num:
                population = random.sample(loaded_all_agents, num)
                print(f"Selected {len(population)} agents (requested {num}).")
            else:
                population = loaded_all_agents
            return population
        except Exception as e:
            print(f"Error loading '{filepath_sim_population}': {e}. Falling back to individual files.")

    # Strategy 2: Fallback to loading individual .pkl files (previous method)
    print(f"Loading agents from individual files in '{directory}'...")
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return []
    
    all_loaded_agents_individual = []
    for filename in os.listdir(directory):
        # Exclude the consolidated 'sim_population.pkl' if it exists in the directory
        if filename.endswith(".pkl") and filename != "sim_population.pkl":
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as f:
                    agent = pickle.load(f)
                    if not hasattr(agent, 'energy'):
                        agent.energy = INITIAL_AGENT_ENERGY # Default energy for compatibility
                    all_loaded_agents_individual.append(agent)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Apply num filtering if requested for individually loaded agents
    if num is not None and len(all_loaded_agents_individual) > num:
        population = random.sample(all_loaded_agents_individual, num)
        print(f"Loaded {len(population)} agents (selected {num} from {len(all_loaded_agents_individual)} available).")
    else:
        population = all_loaded_agents_individual
    
    print(f"Loaded {len(population)} agents.")
    return population

def spawn_food(world, current_food_positions):
    """Spawns a single food item at a random empty location in the air."""
    if len(current_food_positions) >= MAX_FOOD_ON_MAP:
        return current_food_positions

    spawn_success = False
    attempts = 0
    max_attempts = world.width * 2 # Try a few times to find an empty spot

    while not spawn_success and attempts < max_attempts:
        food_x = random.randint(0, world.width - 1)
        # Place food in the air, e.g., at y=2
        food_y = 2

        if world.get_tile(food_x, food_y) == Tile.EMPTY and (food_x, food_y) not in current_food_positions:
            world.place_tile(food_x, food_y, Tile.FOOD)
            current_food_positions.append((food_x, food_y)) # Add to list immediately
            # print(f"Food spawned at ({food_x}, {food_y})")
            spawn_success = True
            # REMOVED: world.apply_gravity() should be called in the main simulation loop
        attempts += 1
    return current_food_positions

def place_agents_in_world(world, agents_list):
    """Places a list of agents into the world grid at valid starting positions."""
    for agent in agents_list:
        agent_placed = False
        attempts = 0
        max_attempts_for_spawn = WORLD_WIDTH * (WORLD_HEIGHT // 2) # Max attempts to find an empty spot in upper half

        # Strategy 1: Try to place at y=0 (top) first
        while not agent_placed and attempts < WORLD_WIDTH * 2: # Give it a few tries at y=0
            current_x = random.randint(0, WORLD_WIDTH - 1)
            if world.get_tile(current_x, 0) == Tile.EMPTY:
                world.agent_pos = (current_x, 0)
                world.place_tile(world.agent_pos[0], world.agent_pos[1], Tile.AGENT)
                agent.position = world.agent_pos
                agent_placed = True
            attempts += 1

        # Strategy 2: If y=0 is consistently blocked, search for any empty spot in the upper half of the world
        if not agent_placed:
            print(f"Warning: Failed to place {agent.name} at y=0 after many attempts. Searching for any empty spot above terrain.")
            attempts = 0
            while not agent_placed and attempts < max_attempts_for_spawn:
                current_x = random.randint(0, WORLD_WIDTH - 1)
                # Try placing in rows from 0 up to WORLD_HEIGHT - 3 (above the ground and first block layer)
                current_y = random.randint(0, WORLD_HEIGHT - 3) 
                if world.get_tile(current_x, current_y) == Tile.EMPTY:
                    world.agent_pos = (current_x, current_y)
                    world.place_tile(world.agent_pos[0], world.agent_pos[1], Tile.AGENT)
                    agent.position = world.agent_pos
                    agent_placed = True
                attempts += 1
        
        # Strategy 3: Critical Fallback - Force place on ground level if no empty spot found anywhere above ground
        if not agent_placed:
            print(f"CRITICAL WARNING: {agent.name} could not be placed on an EMPTY tile anywhere after many attempts. Placing forcefully on default ground.")
            agent_start_x = random.randint(0, world.width - 1)
            world.agent_pos = (agent_start_x, world.height - 2) # Place on the "ground" layer
            world.place_tile(world.agent_pos[0], world.agent_pos[1], Tile.AGENT) # This might overwrite a block
            agent.position = world.agent_pos

        # Reset energy for newly placed/resurrected agents
        agent.energy = INITIAL_AGENT_ENERGY # Ensure they start with full energy

# --- Main Simulation Loop ---
def run_multi_agent_simulation(num=50): # Added num parameter
    # Load the trained population
    population = load_population(SAVE_DIRECTORY, num=num) # Pass num to load_population
    res_list = []

    if not population:
        print("No agents loaded. Please run PretrainingLoop.py first to train and save agents.")
        return

    # Initialize the world for the multi-agent simulation
    world = GravityWorld(width=WORLD_WIDTH, height=WORLD_HEIGHT)
    active_food_positions = [] # Track current food items
    world_states = [] # To store grid states for animation
    population_at_each_step = [] # To store population size for animation title
    num_food_at_each_step = [] # To store food count for animation title

    # Initial placement of agents in the world
    place_agents_in_world(world, population) # Use the new helper function

    print("\nStarting multi-agent simulation...")
    generation = 1 # Start a generation counter
    
    for step in range(NUM_SIMULATION_STEPS):
        if step % 100 == 0:
            print(f"\n--- Simulation Step {step + 1}/{NUM_SIMULATION_STEPS} (Generation {generation}) ---")

        # Periodically spawn food
        if step % FOOD_SPAWN_INTERVAL == 0:
            active_food_positions = spawn_food(world, active_food_positions)

        agents_to_remove = []
        new_agents = []

        # Clear all agent positions from the grid before processing individual actions and gravity
        for y in range(world.height):
            for x in range(world.width):
                if world.grid[y][x] == Tile.AGENT:
                    world.grid[y][x] = Tile.EMPTY

        # Process each agent's turn
        for agent in population:
            if agent.energy <= 0:
                agents_to_remove.append(agent)
                continue # Skip dead agents

            # Energy depletion
            agent.energy -= ENERGY_DEPLETION_RATE
            if agent.energy <= 0:
                #print(f"{agent.name} ran out of energy and died.") # Suppress for cleaner output
                agents_to_remove.append(agent)
                continue

            # 1. Determine Input for Agent
            if world.food_pos != (None, None): # Only provide input if main food hasn't been collected by any agent
                # Find the closest food for this agent
                closest_food_x = None
                min_dist = float('inf')
                
                # Check main food first
                if world.food_pos != (None, None) and world.get_tile(*world.food_pos) == Tile.FOOD:
                    dist_to_main_food = abs(world.food_pos[0] - agent.position[0])
                    if dist_to_main_food < min_dist:
                        min_dist = dist_to_main_food
                        closest_food_x = world.food_pos[0]

                # Check other spawned food items
                for food_pos in active_food_positions:
                    dist = abs(food_pos[0] - agent.position[0])
                    if dist < min_dist:
                        min_dist = dist
                        closest_food_x = food_pos[0]

                if closest_food_x is not None:
                    relative_x_to_closest_food = closest_food_x - agent.position[0]
                else: # No food on map, provide no specific directional input
                    relative_x_to_closest_food = 0
            else: # No food on map
                relative_x_to_closest_food = 0

            abs_distance_to_food = abs(relative_x_to_closest_food)
            if abs_distance_to_food == 0:
                scaled_input = 0.0
            else:
                scaled_input = np.sign(relative_x_to_closest_food) * (1.0 / (abs_distance_to_food + 1.0))
            
            noisy_input = scaled_input + random.gauss(0, INPUT_NOISE_STD)
            agent.receive_inputs([noisy_input])

            # 2. Agent decides action
            agent.step()
            action = agent.decide_action()

            # Set world's agent_pos to current agent's position for its movement
            # This is critical as GravityWorld's move_agent/jump still rely on world.agent_pos
            world.agent_pos = agent.position 

            # 3. Execute Action in World (updates world.agent_pos and places Tile.AGENT temporarily)
            if action == 'jump_left':
                world.jump(-1)
            elif action == 'left':
                world.move_agent(-1)
            elif action == 'right':
                world.move_agent(1)
            elif action == 'jump_right':
                world.jump(1)
            elif action == 'wait':
                pass # Waiting is always "successful", no world change

            # Update agent's internal position from world's updated position after its move/jump
            agent.position = world.agent_pos
            
            # Re-place the agent in the world grid at its new position after its individual action
            # This is important so world.apply_gravity sees all agents in their post-action spots
            world.place_tile(agent.position[0], agent.position[1], Tile.AGENT)


        # Apply gravity after all agents have taken their actions
        # Pass lists of positions for global gravity application
        current_agent_positions_on_grid = [(a.position[0], a.position[1]) for a in population if a.energy > 0 and a.position is not None]
        updated_agent_positions, updated_food_positions = world.apply_gravity(current_agent_positions_on_grid, active_food_positions)

        # NEW: Re-synchronize agent.position for ALL agents by mapping to updated positions
        # The order of updated_agent_positions corresponds to the order they were collected in current_agent_positions_on_grid
        agent_index = 0
        for agent in population:
            if agent.energy > 0 and agent.position is not None:
                if agent_index < len(updated_agent_positions):
                    agent.position = updated_agent_positions[agent_index]
                agent_index += 1

        # Update active_food_positions based on gravity's outcome
        active_food_positions = updated_food_positions

        # Check for food collection and apply reward
        food_to_remove_from_list_current_step = []
        for agent in population:
            # Check main food (world.food_pos)
            if world.food_pos != (None, None) and agent.position == world.food_pos:
                agent.energy += FOOD_ENERGY_RECHARGE
                world.food_pos = (None, None) # Remove main food
                print(f"{agent.name} collected main food! Energy: {agent.energy:.2f}")

            # Check other spawned food (using the updated active_food_positions list)
            for food_pos in active_food_positions:
                if agent.position == food_pos:
                    agent.energy += FOOD_ENERGY_RECHARGE
                    world.place_tile(food_pos[0], food_pos[1], Tile.EMPTY) # Clear food from grid
                    food_to_remove_from_list_current_step.append(food_pos)
                    print(f"{agent.name} collected spawned food! Energy: {agent.energy:.2f}")
                    
        # Remove collected food from active_food_positions list (must remove after iteration completes)
        for f_pos_to_remove in food_to_remove_from_list_current_step:
            if f_pos_to_remove in active_food_positions:
                active_food_positions.remove(f_pos_to_remove)


        # Reproduction logic
        for agent in population: # Iterate over the original population list
            if agent.energy >= REPRODUCTION_ENERGY_THRESHOLD:
                child_agent = mutate_agent(agent, MUTATION_RATE, MUTATION_STRENGTH)
                #child_agent.name = f"{agent.name}_child{random.randint(0,99)}" # Name handled in mutate_agent now
                child_agent.energy = INITIAL_AGENT_ENERGY # Child starts with fresh energy
                agent.energy -= REPRODUCTION_COST # Parent pays energy cost
                new_agents.append(child_agent)
                print(f"{agent.name} reproduced! Created {child_agent.name}. Parent energy: {agent.energy:.2f}")

        # Remove dead agents from population (after all processing)
        for dead_agent in agents_to_remove:
            res_list.append(dead_agent)
            while len(res_list) > 10:
                res_list.pop(0)
            if dead_agent.position is not None:
                world.place_tile(dead_agent.position[0], dead_agent.position[1], Tile.EMPTY) # Clear from world
            population.remove(dead_agent)
            #print(f"Removed {dead_agent.name} (died). Population size: {len(population)}") # Suppress for cleaner output


        # Add new agents to population
        for new_agent in new_agents:
            population.append(new_agent)
            # Place new agents at top of world similar to initial placement
            # This will also reset their energy to INITIAL_AGENT_ENERGY
            place_agents_in_world(world, [new_agent]) # Use helper for single new agent


        # Check for population extinction and resurrection
        if not population:
            print(f"\n--- Generation {generation} extinct. Resurrecting new population! ---")
            generation += 1

            resurrected_population = []
            for agent_to_res in res_list: # Resurrect 'num' agents
                
                
                # Apply mutation to this new agent to create variation
                # Using stronger mutation to encourage more exploration when resurrecting
                resurrected_agent = mutate_agent(agent_to_res, MUTATION_RATE, MUTATION_STRENGTH)
                resurrected_population.append(resurrected_agent)

            population = resurrected_population
            
            # Clear food and re-spawn for the new generation
            # Remove all food tiles from the grid
            for f_pos in active_food_positions:
                world.place_tile(f_pos[0], f_pos[1], Tile.EMPTY)
            world.food_pos = (None, None) # Clear main food too
            active_food_positions = [] # Clear the list
            
            # Place the resurrected agents in the world (resets their energy as well)
            place_agents_in_world(world, population)

            # Spawn initial food for the new generation
            active_food_positions = spawn_food(world, active_food_positions)
            
            print(f"Resurrection complete. New population size: {len(population)}")


        # Capture the current world state and metrics for animation
        world_states.append(world.render())
        population_at_each_step.append(len(population))
        num_food_at_each_step.append(len(active_food_positions))

    save_path = os.path.join(SAVE_DIRECTORY, f"sim_population.pkl")
    if len(population) > 0:
        with open(save_path, 'wb') as f:
            pickle.dump(population, f)
        print(f"Saved {population} to {save_path}")

    print("\nMulti-agent simulation finished.")
    print(f"Final Population size: {len(population)}")

    print("Generating animation...")
    fig, ax = plt.subplots(figsize=(WORLD_WIDTH / 5, WORLD_HEIGHT / 5)) # Adjust figsize as needed
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow(world_states[0], cmap=cmap, norm=norm, origin='upper')

    def update(frame):
        img.set_array(world_states[frame])
        ax.set_title(f"Step: {frame+1}, Agents: {population_at_each_step[frame]}, Food: {num_food_at_each_step[frame]}")
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(world_states), interval=100, blit=True) # interval in ms

    animation_filename = "simulation_animation.gif" # or .mp4, but requires ffmpeg
    print(f"Saving animation to {animation_filename} (this may take a while)...")
    ani.save(animation_filename, writer='pillow', fps=10) # 'pillow' for GIF, 'ffmpeg' for MP4 (requires ffmpeg installed)
    plt.close(fig) # Close the figure to free up memory
    print("Animation saved!")


if __name__ == "__main__":
    # Ensure the 'trained_agents' directory exists from the previous step
    if not os.path.exists(SAVE_DIRECTORY):
        print(f"Directory '{SAVE_DIRECTORY}' not found.")
        print("Please run PretrainingLoop.py first to train and save agents.")
    else:
        run_multi_agent_simulation()

if __name__ == "__main__":
    # Ensure the 'trained_agents' directory exists from the previous step
    if not os.path.exists(SAVE_DIRECTORY):
        print(f"Directory '{SAVE_DIRECTORY}' not found.")
        print("Please run PretrainingLoop.py first to train and save agents.")
    else:
        run_multi_agent_simulation(num=10)