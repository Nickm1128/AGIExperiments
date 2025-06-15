from Agent import Agent
from GravityWorld import GravityWorld, Tile
import random 

# --- Training Loop Setup ---
def setup_training_loop():
    # Training parameters
    NUM_EPISODES = 1000
    STEPS_PER_EPISODE = 50
    INPUT_NOISE_STD = 0.1 # Standard deviation for Gaussian noise

    # Initialize a single agent for training
    agent = Agent("TrainedAgent", neuron_count=50) # Using 10 neurons, so last 4 are output

    print("Starting training loop...")
    for episode in range(NUM_EPISODES):
        world = GravityWorld(width=15, height=10) # Create a new world for each episode

        # Place agent and food randomly on the ground (just above the solid block layer)
        agent_start_x = random.randint(0, world.width - 1)
        food_start_x = random.randint(0, world.width - 1)

        # Ensure they are on the second to last row (above the ground blocks)
        world.agent_pos = (agent_start_x, world.height - 2)
        world.food_pos = (food_start_x, world.height - 2)

        world.place_tile(world.agent_pos[0], world.agent_pos[1], Tile.AGENT)
        world.place_tile(world.food_pos[0], world.food_pos[1], Tile.FOOD)


        agent.reset()
        previous_distance_to_food = abs(world.agent_pos[0] - world.food_pos[0])

        print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")
        print(f"Initial: Agent at {world.agent_pos}, Food at {world.food_pos}")

        for step in range(STEPS_PER_EPISODE):
            if world.is_food_collected():
                print(f"Food collected in {step} steps!")
                break

            # 1. Determine Input for Agent (relative horizontal distance to food with noise)
            if world.food_pos != (None, None):
                relative_x_to_food = world.food_pos[0] - world.agent_pos[0]
            else: # Food collected, provide no specific directional input
                relative_x_to_food = 0

            # Add noise to input
            noisy_input = relative_x_to_food + random.gauss(0, INPUT_NOISE_STD)
            agent.receive_inputs([noisy_input])

            # 2. Agent decides action
            agent.step() # Agent thinks
            action = agent.decide_action()

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

            agent.learn(reward)
            # print(f"Step {step+1}: Action: {action}, Agent pos: {world.agent_pos}, Food pos: {world.food_pos}, Reward: {reward:.2f}")
        if not world.is_food_collected():
            print(f"Episode {episode + 1} finished without collecting food. Final Agent pos: {world.agent_pos}.")
        else:
            print(f"Episode {episode + 1} completed. Final Agent pos: {world.agent_pos}, Food pos: {world.food_pos}")

    print("\nTraining loop finished.")

# Run the training loop
if __name__ == "__main__":
    setup_training_loop()