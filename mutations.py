import random
import copy
import Agent
from Agent import Synapse

def mutate_agent(original_agent, mutation_rate, mutation_strength):
    """
    Creates a mutated copy of an agent.
    """
    mutated_agent = copy.deepcopy(original_agent)
    mutated_agent.name = original_agent.name + f"_mut{random.randint(0, 9999)}"

    for syn in mutated_agent.synapses:
        if random.random() < mutation_rate:
            syn.weight += random.uniform(-mutation_strength, mutation_strength)
            syn.weight = max(min(syn.weight, 2.0), -2.0)

    for neuron in mutated_agent.neurons:
        if random.random() < mutation_rate:
            neuron.threshold_pos += random.uniform(-0.1, 0.1)
            neuron.threshold_neg += random.uniform(-0.1, 0.1)
            neuron.threshold_pos = max(0.01, min(1.0, neuron.threshold_pos))
            neuron.threshold_neg = min(-0.01, max(-1.0, neuron.threshold_neg))

    return mutated_agent

def crossover_agents(parent1, parent2, neuron_count):
    """
    Creates a child agent by combining synapses from two parent agents.
    Assumes parents have the same number of neurons and similar synapse structure.
    """
    child_agent = Agent("Crossover_Child", neuron_count)
    child_agent.synapses = [] # Clear initial random synapses for child

    # Simple uniform crossover: for each synapse, randomly pick from parent1 or parent2
    num_synapses = max(len(parent1.synapses), len(parent2.synapses))

    for i in range(num_synapses):
        chosen_synapse_parent = None
        if random.random() < 0.5: # 50% chance to inherit from parent1 or parent2
            if i < len(parent1.synapses):
                chosen_synapse_parent = parent1.synapses[i]
            elif i < len(parent2.synapses): # Fallback if parent1 has fewer synapses
                chosen_synapse_parent = parent2.synapses[i]
        else:
            if i < len(parent2.synapses):
                chosen_synapse_parent = parent2.synapses[i]
            elif i < len(parent1.synapses): # Fallback if parent2 has fewer synapses
                chosen_synapse_parent = parent1.synapses[i]

        if chosen_synapse_parent:
            pre_neuron_index = 0
            post_neuron_index = 0

            if chosen_synapse_parent in parent1.synapses:
                pre_neuron_index = parent1.neurons.index(chosen_synapse_parent.pre)
                post_neuron_index = parent1.neurons.index(chosen_synapse_parent.post)
            elif chosen_synapse_parent in parent2.synapses:
                pre_neuron_index = parent2.neurons.index(chosen_synapse_parent.pre)
                post_neuron_index = parent2.neurons.index(chosen_synapse_parent.post)
            else:
                print("Warning: Chosen synapse parent not found in either parent's synapse list during crossover.")
                continue # Skip this synapse

            pre_neuron_child = child_agent.neurons[pre_neuron_index]
            post_neuron_child = child_agent.neurons[post_neuron_index]

            child_agent.synapses.append(Synapse(pre_neuron_child, post_neuron_child, chosen_synapse_parent.weight))

    while len(child_agent.synapses) < len(parent1.synapses): 
        pre, post = random.sample(child_agent.neurons, 2)
        weight = random.uniform(-1, 1)
        child_agent.synapses.append(Synapse(pre, post, weight))

    return child_agent