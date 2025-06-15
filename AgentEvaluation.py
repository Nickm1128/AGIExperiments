import os
import pickle
import random
import numpy as np

# Import classes and configurations from your existing modules
from Agent import Agent
from GravityWorld import GravityWorld, Tile
from multiagentsimulation import load_population, spawn_food # Reuse helper functions
from multiagentsimulation import (
    SAVE_DIRECTORY, WORLD_WIDTH, WORLD_HEIGHT,
    INITIAL_AGENT_ENERGY, ENERGY_DEPLETION_RATE, FOOD_ENERGY_RECHARGE,
    INPUT_NOISE_STD, NEURON_COUNT
) 

# --- Evaluation Parameters ---
NUM_EVALUATION_EPISODES = 50 # How many test scenarios each agent runs through
STEPS_PER_EVALUATION_EPISODE = 101 # Max steps per evaluation episode
EVALUATION_FOOD_SPAWN_INTERVAL = 25 # Food spawning interval during evaluation
EVALUATION_MAX_FOOD_ON_MAP = 3 # Max food on map during evaluation

# --- Evaluation Function for a Single Agent ---
def evaluate_agent_performance(agent, evaluation_world_seed=None):
    """
    Evaluates the performance of a single agent over multiple episodes.
    Args:
        agent: The Agent object to evaluate.
        evaluation_world_seed: Optional seed to ensure consistent world generation for evaluation.
    Returns:
        A dictionary of evaluation metrics for the agent.
    """
    print(f"\n--- Evaluating Agent: {agent.name} ---")

    total_food_collected = 0
    total_steps_survived = 0
    successful_episodes = 0
    total_steps_to_collect_food = [] # Stores steps taken for each food collection

    for episode in range(NUM_EVALUATION_EPISODES):
        if evaluation_world_seed is not None:
            random.seed(evaluation_world_seed + episode) # Use seed for consistent worlds
        
        # Create a new world for each episode
        world = GravityWorld(width=WORLD_WIDTH, height=WORLD_HEIGHT)
        current_agent_eval_pos = None # Track agent's position within evaluation world
        active_food_positions = [] # Track food for this evaluation episode

        # Place agent at the top of the world on an empty spot for evaluation
        agent_placed = False
        attempts = 0
        max_attempts_for_spawn = WORLD_WIDTH * (WORLD_HEIGHT // 2) 
        while not agent_placed and attempts < WORLD_WIDTH * 2: # Give it a few tries at y=0
            current_x = random.randint(0, WORLD_WIDTH - 1)
            if world.get_tile(current_x, 0) == Tile.EMPTY:
                world.place_tile(current_x, 0, Tile.AGENT)
                current_agent_eval_pos = (current_x, 0)
                agent_placed = True
            attempts += 1
        if not agent_placed:
            # Fallback for evaluation if top is consistently blocked
            attempts = 0
            while not agent_placed and attempts < max_attempts_for_spawn:
                current_x = random.randint(0, WORLD_WIDTH - 1)
                current_y = random.randint(0, WORLD_HEIGHT - 3) 
                if world.get_tile(current_x, current_y) == Tile.EMPTY:
                    world.place_tile(current_x, current_y, Tile.AGENT)
                    current_agent_eval_pos = (current_x, current_y)
                    agent_placed = True
                attempts += 1
            if not agent_placed:
                print(f"CRITICAL WARNING (Evaluation): {agent.name} could not be placed on an EMPTY tile. Placing forcefully.")
                current_agent_eval_pos = (random.randint(0, world.width - 1), world.height - 2)
                world.place_tile(*current_agent_eval_pos, Tile.AGENT)

        agent.reset() # Reset agent's internal state (neuron states)
        agent.energy = INITIAL_AGENT_ENERGY # Reset energy for evaluation episode
        
        food_collected_this_episode = 0
        steps_this_episode = 0
        agent_alive_this_episode = True

        for step in range(STEPS_PER_EVALUATION_EPISODE):
            if not agent_alive_this_episode:
                break # Agent died in this episode

            # Apply gravity to all entities on the grid
            # Need agent's current position to apply gravity correctly
            # Temporarily set world.agent_pos for gravity logic to work on this agent
            world.agent_pos = current_agent_eval_pos
            updated_agent_pos_list, updated_food_pos_list = world.apply_gravity([current_agent_eval_pos], active_food_positions)
            
            # Update current_agent_eval_pos and active_food_positions from returned lists
            if updated_agent_pos_list: # Should contain only one element for this agent
                current_agent_eval_pos = updated_agent_pos_list[0]
            active_food_positions = updated_food_pos_list

            # Energy depletion
            agent.energy -= ENERGY_DEPLETION_RATE
            if agent.energy <= 0:
                # print(f"Agent {agent.name} died in evaluation episode {episode + 1}.")
                agent_alive_this_episode = False
                break

            steps_this_episode += 1

            # Periodically spawn food for evaluation
            if step % EVALUATION_FOOD_SPAWN_INTERVAL == 0:
                active_food_positions = spawn_food(world, active_food_positions)
            
            # Determine Input for Agent using local tile view (same as multi-agent simulation)
            radius = 15
            local_inputs = []
            ax, ay = current_agent_eval_pos
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

            agent.receive_inputs(local_inputs)

            # Agent decides action
            agent.step()
            action = agent.decide_action()

            # Execute Action in World (temporarily set world.agent_pos for movement methods)
            world.agent_pos = current_agent_eval_pos
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

            # Update agent's position from world after its action
            current_agent_eval_pos = world.agent_pos

            # Check for food collection
            food_to_remove_from_list_this_step = []
            for food_pos in active_food_positions:
                if current_agent_eval_pos == food_pos:
                    agent.energy += FOOD_ENERGY_RECHARGE
                    world.place_tile(food_pos[0], food_pos[1], Tile.EMPTY) # Clear food from grid
                    food_to_remove_from_list_this_step.append(food_pos)
                    food_collected_this_episode += 1
                    total_steps_to_collect_food.append(steps_this_episode) # Record steps taken to collect this food
            
            for f_pos_to_remove in food_to_remove_from_list_this_step:
                if f_pos_to_remove in active_food_positions:
                    active_food_positions.remove(f_pos_to_remove)

        total_food_collected += food_collected_this_episode
        total_steps_survived += steps_this_episode
        if food_collected_this_episode > 0: # If at least one food was collected in this episode
            successful_episodes += 1

    # Calculate average metrics
    avg_food_collected_per_episode = total_food_collected / NUM_EVALUATION_EPISODES
    avg_steps_survived_per_episode = total_steps_survived / NUM_EVALUATION_EPISODES
    episode_success_rate_percent = (successful_episodes / NUM_EVALUATION_EPISODES) * 100
    
    avg_steps_per_food_collected = 0
    if total_steps_to_collect_food:
        avg_steps_per_food_collected = sum(total_steps_to_collect_food) / len(total_steps_to_collect_food)

    print(f"  Avg Food Collected/Episode: {avg_food_collected_per_episode:.2f}")
    print(f"  Avg Steps Survived/Episode: {avg_steps_survived_per_episode:.2f}")
    print(f"  Episode Success Rate: {episode_success_rate_percent:.2f}%")
    print(f"  Avg Steps per Food Collected: {avg_steps_per_food_collected:.2f}")

    return {
        "name": agent.name,
        "avg_food_collected": avg_food_collected_per_episode,
        "avg_steps_survived": avg_steps_survived_per_episode,
        "episode_success_rate": episode_success_rate_percent,
        "avg_steps_per_food_collected": avg_steps_per_food_collected
    }

# --- Main Evaluation Script Execution ---
if __name__ == "__main__":
    # Ensure a consistent random state for world generation during evaluation
    EVALUATION_MASTER_SEED = 42
    random.seed(EVALUATION_MASTER_SEED)
    np.random.seed(EVALUATION_MASTER_SEED) # For numpy random calls if any in world generation

    print("Starting agent population evaluation...")

    # Load the trained population
    # You might want to load the final population from a completed multi-agent simulation run.
    # For example, if you saved 'final_population.pkl'
    # population_to_evaluate = load_population(SAVE_DIRECTORY, filename="final_population.pkl")
    population_to_evaluate = load_population(SAVE_DIRECTORY, num=50)

    if not population_to_evaluate:
        print("No agents loaded for evaluation. Please ensure agents are trained and saved.")
    else:
        all_evaluation_results = []
        for agent in population_to_evaluate:
            results = evaluate_agent_performance(agent, evaluation_world_seed=EVALUATION_MASTER_SEED)
            all_evaluation_results.append(results)

        # Summarize overall population performance
        print("\n--- Overall Population Evaluation Summary ---")
        if all_evaluation_results:
            avg_food = np.mean([r["avg_food_collected"] for r in all_evaluation_results])
            avg_survival = np.mean([r["avg_steps_survived"] for r in all_evaluation_results])
            avg_success = np.mean([r["episode_success_rate"] for r in all_evaluation_results])
            avg_steps_per_food = np.mean([r["avg_steps_per_food_collected"] for r in all_evaluation_results if r["avg_steps_per_food_collected"] > 0]) # Exclude 0 to avoid division by zero if no food collected

            print(f"Population Average Food Collected/Episode: {avg_food:.2f}")
            print(f"Population Average Steps Survived/Episode: {avg_survival:.2f}")
            print(f"Population Average Episode Success Rate: {avg_success:.2f}%")
            if avg_steps_per_food:
                print(f"Population Average Steps per Food Collected (successful collections): {avg_steps_per_food:.2f}")
            else:
                print("No food collected across population during evaluation.")

            # Optional: Sort and display top performers
            print("\n--- Top Agents by Food Collected ---")
            top_agents = sorted(all_evaluation_results, key=lambda x: x["avg_food_collected"], reverse=True)[:5]
            for i, agent_res in enumerate(top_agents):
                print(f"{i+1}. {agent_res['name']}: {agent_res['avg_food_collected']:.2f} food, {agent_res['episode_success_rate']:.2f}% success")
        else:
            print("No evaluation results to display.")

    print("\nEvaluation finished.")