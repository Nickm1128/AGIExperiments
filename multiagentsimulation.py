import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import colors, cmap, bounds, norm

# Import classes from your existing modules
from Agent import Agent
from GravityWorld import GravityWorld, Tile
from mutations import mutate_agent

# --- Configuration Parameters ---
SAVE_DIRECTORY = "trained_agents"
WORLD_WIDTH = 100
WORLD_HEIGHT = 20
INITIAL_WORLD_WIDTH = 100
INITIAL_WORLD_HEIGHT = 20
INITIAL_AGENT_ENERGY = 100.0
ENERGY_DEPLETION_RATE = 1.5 # Energy lost per simulation step
FOOD_ENERGY_RECHARGE = 100.0 # Energy gained from eating food
FOOD_SPAWN_INTERVAL = 1 # Every N steps, try to spawn food
MAX_FOOD_ON_MAP = 50 # Maximum number of food items allowed at once
REPRODUCTION_ENERGY_THRESHOLD = 101.0 # Energy needed to reproduce
REPRODUCTION_COST = 20.0 # Energy cost of reproduction
MUTATION_RATE = .5 # Rate for mutations during reproduction
MUTATION_STRENGTH = .5 # Strength for mutations during reproduction
NEURON_COUNT = 50 # Consistent neuron count for agents
INPUT_NOISE_STD = .1
MAX_WORLD_WIDTH = 500
MAX_WORLD_HEIGHT = 20

NUM_SIMULATION_STEPS = 10_000_000 # Total steps for the multi-agent simulation

# New constants for population control
MAX_POPULATION_SIZE = 50
OVERPOPULATION_THRESHOLD = 75
RES_LIST_LIMIT = 50 # Limit for res_list size before applying selection for dead agents


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
        max_attempts_for_spawn = world.width * (world.height // 2)  # Max attempts to find an empty spot in upper half

        # Strategy 1: Try to place at y=0 (top) first
        while not agent_placed and attempts < world.width * 2:  # Give it a few tries at y=0
            current_x = random.randint(0, world.width - 1)
            if world.get_tile(current_x, 0) == Tile.EMPTY:
                world.place_tile(current_x, 0, Tile.AGENT)
                agent.position = (current_x, 0)
                agent_placed = True
            attempts += 1

        # Strategy 2: If y=0 is consistently blocked, search for any empty spot in the upper half of the world
        if not agent_placed:
            print(f"Warning: Failed to place {agent.name} at y=0 after many attempts. Searching for any empty spot above terrain.")
            attempts = 0
            while not agent_placed and attempts < max_attempts_for_spawn:
                current_x = random.randint(0, world.width - 1)
                # Try placing in rows from 0 up to world.height - 3 (above the ground and first block layer)
                current_y = random.randint(0, world.height - 3)
                if world.get_tile(current_x, current_y) == Tile.EMPTY:
                    world.place_tile(current_x, current_y, Tile.AGENT)
                    agent.position = (current_x, current_y)
                    agent_placed = True
                attempts += 1
        
        # Strategy 3: Critical Fallback - Force place on ground level if no empty spot found anywhere above ground
        if not agent_placed:
            print(f"CRITICAL WARNING: {agent.name} could not be placed on an EMPTY tile anywhere after many attempts. Placing forcefully on default ground.")
            agent_start_x = random.randint(0, world.width - 1)
            agent_start_y = world.height - 2 # Place on the "ground" layer
            world.place_tile(agent_start_x, agent_start_y, Tile.AGENT)
            agent.position = (agent_start_x, agent_start_y)

        # Reset energy for newly placed/resurrected agents
        agent.energy = INITIAL_AGENT_ENERGY # Ensure they start with full energy


# --- Main Simulation Loop ---
def run_multi_agent_simulation(num=50):
    population = load_population(SAVE_DIRECTORY, num=num)
    res_list = []

    if not population:
        print("No agents loaded. Please run PretrainingLoop.py first to train and save agents.")
        return

    current_width = INITIAL_WORLD_WIDTH
    current_height = INITIAL_WORLD_HEIGHT
    world = GravityWorld(width=current_width, height=current_height)

    # Start with a flat terrain: only bottom row is BLOCK
    for x in range(current_width):
        world.place_tile(x, current_height - 1, Tile.BLOCK)

    active_food_positions = []
    world_states = []
    population_at_each_step = []
    num_food_at_each_step = []

    place_agents_in_world(world, population)
    print("\nStarting multi-agent simulation...")
    generation = 1

    for step in range(NUM_SIMULATION_STEPS):
        if step % 100 == 0:
            print(f"\n--- Simulation Step {step + 1}/{NUM_SIMULATION_STEPS} (Generation {generation}) (Pop size: {len(population)}) ---")

        # Gradual world expansion
        if step % 800 == 0 and (current_width < MAX_WORLD_WIDTH or current_height < MAX_WORLD_HEIGHT):
            new_width = min(current_width + 4, MAX_WORLD_WIDTH)
            new_height = min(current_height + 4, MAX_WORLD_HEIGHT)
            print(f"[World Resize] Increasing world size to {new_width}x{new_height}")
            new_world = GravityWorld(width=new_width, height=new_height)
            for x in range(current_width):
                for y in range(current_height):
                    new_world.place_tile(x, y, world.get_tile(x, y))
                for y in range(current_height, new_height - 1):
                    new_world.place_tile(x, y, Tile.EMPTY)
            world = new_world
            current_width = new_width
            current_height = new_height
            world.enforce_traversable()
            place_agents_in_world(world, population)

        # Terrain evolution
        if step % 5 == 0:
            for x in range(1, world.width - 1):
                heights = []
                for y in range(world.height - 2, -1, -1):
                    if world.get_tile(x, y) == Tile.BLOCK:
                        heights.append(y)
                    elif heights:
                        break
                column_height = heights[0] if heights else (world.height - 1)
                left_height = next((y for y in range(world.height - 2, -1, -1) if world.get_tile(x - 1, y) == Tile.BLOCK), world.height - 1)
                right_height = next((y for y in range(world.height - 2, -1, -1) if world.get_tile(x + 1, y) == Tile.BLOCK), world.height - 1)
                avg_neighbor_height = (left_height + right_height) // 2
                if abs(column_height - avg_neighbor_height) <= 1:
                    direction = random.choice([-1, 0, 1])
                    target_y = column_height - direction
                    if 0 <= target_y < world.height - 1:
                        if direction == 1:
                            world.place_tile(x, target_y, Tile.BLOCK)
                        elif direction == -1 and world.get_tile(x, column_height) == Tile.BLOCK:
                            world.place_tile(x, column_height, Tile.EMPTY)
            world.enforce_traversable()

        proba = (NUM_SIMULATION_STEPS - step) / NUM_SIMULATION_STEPS
        proba = np.max([proba, .2])
        if step % FOOD_SPAWN_INTERVAL == 0 and np.random.rand() > proba:
            active_food_positions = spawn_food(world, active_food_positions)

        agents_to_remove = []
        new_agents = []
        reproduced_this_step = set()

        for y in range(world.height):
            for x in range(world.width):
                if world.grid[y][x] == Tile.AGENT:
                    world.grid[y][x] = Tile.EMPTY

        for agent in population:
            if agent.energy <= 0:
                if len(res_list) < 50 or agent in reproduced_this_step:
                    res_list.append(agent)
                agents_to_remove.append(agent)
                continue

            agent.energy -= ENERGY_DEPLETION_RATE
            if agent.energy <= 0:
                if len(res_list) < 50 or agent in reproduced_this_step:
                    res_list.append(agent)
                agents_to_remove.append(agent)
                continue

            radius = 15
            local_inputs = []
            ax, ay = agent.position
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tx = ax + dx
                    ty = ay + dy
                    tile_val = Tile.BLOCK if not (0 <= tx < world.width and 0 <= ty < world.height) else world.get_tile(tx, ty)
                    normalized_val = tile_val / 3.0
                    local_inputs.append(normalized_val + random.gauss(0, INPUT_NOISE_STD))

            agent.receive_inputs(local_inputs)
            agent.step()
            action = agent.decide_action()
            world.agent_pos = agent.position

            if action == 'jump_left':
                world.jump(-1)
            elif action == 'left':
                world.move_agent(-1)
            elif action == 'right':
                world.move_agent(1)
            elif action == 'jump_right':
                world.jump(1)

            agent.position = world.agent_pos
            world.place_tile(agent.position[0], agent.position[1], Tile.AGENT)

        current_agent_positions_on_grid = [(a.position[0], a.position[1]) for a in population if a.energy > 0 and a.position is not None]
        updated_agent_positions, updated_food_positions = world.apply_gravity(current_agent_positions_on_grid, active_food_positions)
        world.enforce_traversable()

        agent_index = 0
        for agent in population:
            if agent.energy > 0 and agent.position is not None and agent_index < len(updated_agent_positions):
                agent.position = updated_agent_positions[agent_index]
                agent_index += 1

        active_food_positions = updated_food_positions

        food_to_remove_from_list_current_step = []
        for agent in population:
            if world.food_pos != (None, None) and agent.position == world.food_pos:
                agent.energy += FOOD_ENERGY_RECHARGE
                world.food_pos = (None, None)

            for food_pos in active_food_positions:
                if agent.position == food_pos:
                    agent.energy += FOOD_ENERGY_RECHARGE
                    world.place_tile(food_pos[0], food_pos[1], Tile.EMPTY)
                    food_to_remove_from_list_current_step.append(food_pos)

        for f_pos_to_remove in food_to_remove_from_list_current_step:
            if f_pos_to_remove in active_food_positions:
                active_food_positions.remove(f_pos_to_remove)

        multiplier = ((NUM_SIMULATION_STEPS - step) / NUM_SIMULATION_STEPS * .75) ** 2
        for agent in population:
            if agent.energy >= REPRODUCTION_ENERGY_THRESHOLD:
                child_agent = mutate_agent(agent, MUTATION_RATE * multiplier, MUTATION_STRENGTH * multiplier)
                child_agent.energy = INITIAL_AGENT_ENERGY
                agent.energy -= REPRODUCTION_COST
                new_agents.append(child_agent)
                reproduced_this_step.add(agent)

        for dead_agent in agents_to_remove:
            if dead_agent.position is not None:
                world.place_tile(dead_agent.position[0], dead_agent.position[1], Tile.EMPTY)
            if dead_agent in population:
                population.remove(dead_agent)

        for new_agent in new_agents:
            population.append(new_agent)
            place_agents_in_world(world, [new_agent])

        if len(population) > OVERPOPULATION_THRESHOLD:
            print(f"Population {len(population)} exceeds {OVERPOPULATION_THRESHOLD}. Pruning...")
            population.sort(key=lambda agent: agent.energy, reverse=True)
            agents_to_prune_due_to_overpopulation = population[MAX_POPULATION_SIZE:]
            for pruned_agent in agents_to_prune_due_to_overpopulation:
                if len(res_list) < 50 or pruned_agent in reproduced_this_step:
                    res_list.append(pruned_agent)
                if pruned_agent.position is not None:
                    world.place_tile(pruned_agent.position[0], pruned_agent.position[1], Tile.EMPTY)
            population = population[:MAX_POPULATION_SIZE]

        if not population:
            print(f"\n--- Generation {generation} extinct. Resurrecting new population! ---")
            generation += 1

            resurrected_population = []
            num_to_resurrect = num

            if len(res_list) < num_to_resurrect:
                print(f"Warning: Not enough agents in res_list ({len(res_list)}) for full resurrection ({num_to_resurrect}). Filling with new random agents.")
                for i in range(num_to_resurrect - len(res_list)):
                    new_rand_agent = Agent(f"NewRandomAgent_{generation}_{i+1}", NEURON_COUNT)
                    res_list.append(new_rand_agent)

            agents_for_resurrection_base = res_list[-num_to_resurrect:]
            for agent_to_res_base in agents_for_resurrection_base:
                resurrected_agent = mutate_agent(agent_to_res_base, MUTATION_RATE * multiplier, MUTATION_STRENGTH * multiplier)
                resurrected_population.append(resurrected_agent)

            res_list = []
            population = resurrected_population
            active_food_positions = []
            place_agents_in_world(world, population)
            active_food_positions = spawn_food(world, active_food_positions)
            print(f"Resurrection complete. New population size: {len(population)}")

        if step >= NUM_SIMULATION_STEPS - 1000:
            world_states.append(world.render())
            population_at_each_step.append(len(population))
            num_food_at_each_step.append(len(active_food_positions))

    save_path = os.path.join(SAVE_DIRECTORY, f"sim_population.pkl")
    if len(population) > 0:
        with open(save_path, 'wb') as f:
            pickle.dump(population, f)
        print(f"Saved population to {save_path}")

    print("\nMulti-agent simulation finished.")
    print(f"Final Population size: {len(population)}")

    print("Generating animation...")
    fig, ax = plt.subplots(figsize=(WORLD_WIDTH / 5, WORLD_HEIGHT / 5))
    ax.set_xticks([])
    ax.set_yticks([])

    img = ax.imshow(world_states[0], cmap=cmap, norm=norm, origin='upper')

    def update(frame):
        img.set_array(world_states[frame])
        ax.set_title(f"Step: {frame+1}, Agents: {population_at_each_step[frame]}, Food: {num_food_at_each_step[frame]}")
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(world_states), interval=100, blit=True)
    animation_filename = "simulation_animation.gif"
    print(f"Saving animation to {animation_filename} (this may take a while)...")
    ani.save(animation_filename, writer='pillow', fps=5)
    plt.close(fig)
    print("Animation saved!")

if __name__ == "__main__":
    # Ensure the 'trained_agents' directory exists from the previous step
    if not os.path.exists(SAVE_DIRECTORY):
        print(f"Directory '{SAVE_DIRECTORY}' not found.")
        print("Please run PretrainingLoop.py first to train and save agents.")
    else:
        run_multi_agent_simulation()
