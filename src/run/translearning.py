from pathlib import Path
import sys

for i in range(3):
    path_root = Path(__file__).parents[i]
    sys.path.append(str(path_root))

import copy
import json
import os
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from my_run import init_learner, agent_evaluate_sequential, agent_train_sequential, conv_args
from params import AgentParam
from utils import logging
from pareto import cal_all_pareto_frontier
from run.ea import EA, EAPopulation

_log = logging.get_logger()
logger = logging.Logger(_log)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransferLearningEA:
    def __init__(self, source_ea, target_args, mutation_rate, crossover_rate, pop_size, encoder, decoder, encoder_optimizer):
        self.source_ea = source_ea
        self.target_args = target_args
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.pop_size = pop_size
        self.target_populations = []
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.loss_fn = nn.MSELoss()

    def train_encoder(self, source_states, target_states, epochs=100):
        for epoch in range(epochs):
            self.encoder_optimizer.zero_grad()
            encoded_states = self.encoder(source_states)
            decoded_states = self.decoder(encoded_states)
            loss = self.loss_fn(decoded_states, target_states)
            loss.backward()
            self.encoder_optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    def transfer_knowledge(self):
        # Transfer each individual from the source EA to the target EA
        for source_pop in self.source_ea.populations:
            # Clone the source population
            target_pop = source_pop.clone()
            # Map source states to target states using encoder
            target_pop.agent_param.agent.state_dict().update(
                {k: self.encoder(v) if 'weight' in k else v for k, v in source_pop.agent_param.agent.state_dict().items()}
            )
            # Apply the target environment args
            target_pop.args = self.target_args
            target_pop.train()
            self.target_populations.append(target_pop)

    def initialize_new_population(self):
        # Initialize the target EA with the transferred individuals and random individuals
        ea = EA(self.target_args, self.pop_size, self.mutation_rate, self.crossover_rate)
        # Add the transferred individuals to the new EA
        for pop in self.target_populations:
            ea.populations.append(pop)

        # If the number of transferred individuals is less than the target population size, add random individuals
        while len(ea.populations) < self.pop_size:
            population = EAPopulation(self.target_args)
            population.evaluate()
            ea.populations.append(population)

        return ea

if __name__ == '__main__':
    # Load the source EA
    source_filename = "source_ea.pkl"
    if os.path.exists(source_filename):
        source_ea = EA.load_from_file(source_filename)
    else:
        raise FileNotFoundError(f"Source EA file '{source_filename}' not found!")

    # Load target environment arguments
    target_config = json.load(open("target_running_args.json", "r"))
    target_args = conv_args(target_config, _log)

    # Assuming we know the input, hidden and output dimensions for the encoder
    input_dim = 128  # example dimension
    hidden_dim = 64
    output_dim = 128  # should match target dimension

    # Initialize Encoder and Decoder
    encoder = Encoder(input_dim, hidden_dim, output_dim)
    decoder = Encoder(output_dim, hidden_dim, input_dim)  # Using Encoder as Decoder for simplicity
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    # Sample source and target states for training the encoder
    # Here, we assume we have some source and target state pairs for training purposes
    source_states = torch.randn((100, input_dim))  # Example source states
    target_states = torch.randn((100, output_dim))  # Example target states

    # Initialize TransferLearningEA with source EA, target args, and encoder
    transfer_learning = TransferLearningEA(
        source_ea=source_ea,
        target_args=target_args,
        mutation_rate=0.1,
        crossover_rate=0.1,
        pop_size=10,
        encoder=encoder,
        decoder=decoder,
        encoder_optimizer=encoder_optimizer
    )

    # Train the encoder to map source states to target states
    transfer_learning.train_encoder(source_states, target_states)

    # Transfer knowledge from source EA to target EA
    transfer_learning.transfer_knowledge()

    # Initialize the new EA with transferred individuals
    target_ea = transfer_learning.initialize_new_population()

    # Save the new EA to a file
    target_filename = "target_ea.pkl"
    target_ea.save_to_file(target_filename)

    # Optionally, run evolution on the new EA
    target_ea.evolve(num_generations=10, path=target_filename)
    print(target_ea)
