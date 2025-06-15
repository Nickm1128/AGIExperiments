from Agent import Agent
from GravityWorld import GravityWorld, Tile
import random
import numpy as np
import pickle
import os

# Pull in constants and helpers from the main multi-agent simulation so the
# pretraining environment matches what agents will experience later.
from multiagentsimulation import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    INPUT_NOISE_STD,
    NEURON_COUNT,
    SAVE_DIRECTORY,
    spawn_food,
    place_agents_in_world,
)

def create_initial_population(population_size, neuron_count):
    """
    Generates a list of new Agent instances to form an initial population.
    """
    population = []
    print(f"Generating an initial population of {population_size} agents...")
    for i in range(population_size):
        agent_name = f"Agent_{i+1}_{neuron_count}"
        agent = Agent(agent_name, neuron_count)
        population.append(agent)
    print(f"Successfully created {len(population)} agents.")
    return population

# --- Training Loop Setup ---
def train_single_agent(agent_to_train):
    # Training parameters
    NUM_EPISODES = 100
    STEPS_PER_EPISODE = 50

    print(f"\n--- Starting training for {agent_to_train.name} ---")
    for episode in range(NUM_EPISODES):
        # Create a new world for each episode with procedural terrain
        world = GravityWorld(width=WORLD_WIDTH, height=WORLD_HEIGHT)

        # Place the agent using the same logic as the simulation
        place_agents_in_world(world, [agent_to_train])

        # Spawn a single food item in the air
        active_food_positions = spawn_food(world, [])
        if active_food_positions:
            food_pos = active_food_positions[0]
        else:
            food_pos = (random.randint(0, WORLD_WIDTH - 1), 2)
            world.place_tile(food_pos[0], food_pos[1], Tile.FOOD)
            active_food_positions = [food_pos]


        agent_to_train.reset()
        previous_distance_to_food = abs(agent_to_train.position[0] - food_pos[0]) + abs(agent_to_train.position[1] - food_pos[1])

        for step in range(STEPS_PER_EPISODE):
            if not active_food_positions:
                break

            # 1. Determine Input for Agent using local tile observations
            radius = 15
            local_inputs = []
            ax, ay = agent_to_train.position
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tx = ax + dx
                    ty = ay + dy
                    if 0 <= tx < world.width and 0 <= ty < world.height:
                        tile_val = world.get_tile(tx, ty)
                    else:
                        tile_val = Tile.BLOCK
                    normalized_val = tile_val / 3.0
                    local_inputs.append(normalized_val + random.gauss(0, INPUT_NOISE_STD))

            agent_to_train.receive_inputs(local_inputs)

            # 2. Agent decides action
            agent_to_train.step()
            action = agent_to_train.decide_action()

            # 3. Execute Action in World
            if action == 'jump_left':
                world.jump(-1)
            elif action == 'left':
                world.move_agent(-1)
            elif action == 'right':
                world.move_agent(1)
            elif action == 'jump_right':
                world.jump(1)
            elif action == 'wait':
                pass

            # Apply gravity after agent's action
            updated_agent_pos, updated_food_pos = world.apply_gravity([world.agent_pos], active_food_positions)
            if updated_agent_pos:
                world.agent_pos = updated_agent_pos[0]
                agent_to_train.position = world.agent_pos
            active_food_positions = updated_food_pos

            # 4. Calculate Reward and Agent learns
            reward = -0.05
            collected = False
            for f_pos in list(active_food_positions):
                if world.agent_pos == f_pos:
                    reward += 10.0
                    world.place_tile(f_pos[0], f_pos[1], Tile.EMPTY)
                    active_food_positions.remove(f_pos)
                    collected = True
                    break

            if not collected and active_food_positions:
                fx, fy = active_food_positions[0]
                current_distance_to_food = abs(world.agent_pos[0] - fx) + abs(world.agent_pos[1] - fy)
                if current_distance_to_food < previous_distance_to_food:
                    reward += 1.0
                elif current_distance_to_food > previous_distance_to_food:
                    reward -= 1.0
                else:
                    reward -= 0.1
                previous_distance_to_food = current_distance_to_food

            agent_to_train.learn(reward)
        # Silence output during batch training
        pass
        
    print(f"--- Training for {agent_to_train.name} finished. ---")
    return agent_to_train

if __name__ == "__main__":
    # --- Collect the initial population ---
    POPULATION_SIZE = 50
    # Use the same neuron count as the main simulation
    NEURON_COUNT_FOR_NEW_AGENTS = NEURON_COUNT

    initial_population = create_initial_population(POPULATION_SIZE, NEURON_COUNT_FOR_NEW_AGENTS)

    # --- Train and Save the Population ---
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)  # Create the directory if it doesn't exist

    trained_population = []
    print(f"\nStarting training for {POPULATION_SIZE} agents and saving them to '{SAVE_DIRECTORY}'...")
    for i, agent in enumerate(initial_population):
        print(f"Training agent {i+1}/{POPULATION_SIZE}: {agent.name}")
        trained_agent = train_single_agent(agent)
        trained_population.append(trained_agent)

        # Save the trained agent
        save_path = os.path.join(SAVE_DIRECTORY, f"{trained_agent.name}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(trained_agent, f)
        print(f"Saved {trained_agent.name} to {save_path}")

    print(f"\nAll {len(trained_population)} agents have been trained and saved to the '{SAVE_DIRECTORY}' directory.")
