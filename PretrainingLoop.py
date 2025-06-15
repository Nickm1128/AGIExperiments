from Agent import Agent
from GravityWorld import GravityWorld, Tile
import random 
import numpy as np
import pickle 
import os 

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
    INPUT_NOISE_STD = 0.1 
    WORLD_WIDTH = 50
    WORLD_HEIGHT = 50 

    print(f"\n--- Starting training for {agent_to_train.name} ---")
    for episode in range(NUM_EPISODES):
        # Create a new world for each episode with procedural terrain
        world = GravityWorld(width=WORLD_WIDTH, height=WORLD_HEIGHT)

        # Place agent at the top of the world (y=0) on an EMPTY spot
        agent_placed = False
        y_check = WORLD_HEIGHT
        while not agent_placed:
            agent_start_x = random.randint(0, world.width - 1)
            if world.get_tile(agent_start_x, y_check) == Tile.EMPTY:
                world.agent_pos = (agent_start_x, 0)
                world.place_tile(world.agent_pos[0], world.agent_pos[1], Tile.AGENT)
                agent_placed = True
            else:
                y_check -= 1

        # Place food randomly on the ground level (y = world.height - 2) on an EMPTY spot
        food_placed = False
        y_check = WORLD_HEIGHT
        while not food_placed:
            food_start_x = random.randint(0, world.width - 1)
            if world.get_tile(food_start_x, world.height - (world.height - 3)) == Tile.EMPTY:
                world.food_pos = (food_start_x, world.height - 2)
                world.place_tile(world.food_pos[0], world.food_pos[1], Tile.FOOD)
                food_placed = True


        agent_to_train.reset()
        previous_distance_to_food = abs(world.agent_pos[0] - world.food_pos[0])

        for step in range(STEPS_PER_EPISODE):
            if world.is_food_collected():
                break

            # 1. Determine Input for Agent (relative horizontal distance to food with noise)
            if world.food_pos != (None, None):
                relative_x_to_food = world.food_pos[0] - world.agent_pos[0]
            else: # Food collected, provide no specific directional input
                relative_x_to_food = 0

            # Calculate absolute distance
            abs_distance_to_food = abs(relative_x_to_food)

            # Make input weaker as it gets further away
            if abs_distance_to_food == 0:
                scaled_input = 0.0
            else:
                scaled_input = np.sign(relative_x_to_food) * (1.0 / (abs_distance_to_food + 1.0))

            noisy_input = scaled_input + random.gauss(0, INPUT_NOISE_STD)
            agent_to_train.receive_inputs([noisy_input])

            # 2. Agent decides action
            agent_to_train.step()
            action = agent_to_train.decide_action()

            # 3. Execute Action in World
            action_successful = False
            if action == 'jump_left':
                action_successful = world.jump(-1)
            elif action == 'left':
                action_successful = world.move_agent(-1)
            elif action == 'right':
                action_successful = world.move_agent(1)
            elif action == 'jump_right':
                action_successful = world.jump(1)
            elif action == 'wait':
                action_successful = True # Waiting is always "successful"

            # Apply gravity after agent's action
            world.apply_gravity()

            # 4. Calculate Reward and Agent learns
            reward = 0.0
            if world.is_food_collected():
                reward = 10.0 # High reward for collecting food
            else:
                current_distance_to_food = abs(world.agent_pos[0] - world.food_pos[0])
                if current_distance_to_food < previous_distance_to_food:
                    reward = 1.0 # Positive reward for getting closer
                elif current_distance_to_food > previous_distance_to_food:
                    reward = -1.0 # Negative reward for moving away
                else: # Same distance, or stuck
                    reward = -0.1 # Small penalty for not making progress

                reward -= 0.05 # Small step penalty to encourage efficiency

                previous_distance_to_food = current_distance_to_food

            agent_to_train.learn(reward)
        if not world.is_food_collected():
            pass # Suppress output during batch training
        
    print(f"--- Training for {agent_to_train.name} finished. ---")
    return agent_to_train

if __name__ == "__main__":
    # --- Collect the initial population ---
    POPULATION_SIZE = 50
    NEURON_COUNT_FOR_NEW_AGENTS = 50

    initial_population = create_initial_population(POPULATION_SIZE, NEURON_COUNT_FOR_NEW_AGENTS)

    # --- Train and Save the Population ---
    SAVE_DIRECTORY = "trained_agents"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True) # Create the directory if it doesn't exist

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
