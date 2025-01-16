import random
import os
import numpy as np
import torch
from evaluate import evaluate_HIV, evaluate_HIV_population
from train import MainAgent


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = MainAgent()
    agent.load()
    # Evaluate agent and write score.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=5)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=20)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
