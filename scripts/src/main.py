import math

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random

from scripts.src.metaheuristics_algos.GWO import GWO
from scripts.src.visualizer.ObjectiveFunctionVisualizer import ObjectiveFunctionPlotter

def read_dataset():
  df = pd.read_csv(r"scripts/src/dataset/data_sheep_body_weight.csv")
  return df

def rastrigin(x):
  A = 10
  return A * len(x) + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)

def sphere(x: list[float]) -> float:
  return sum(xi ** 2 for xi in x)

def start_GWO():
  # Space search: 2 dimensions in [-5.12, 5.12]
  space_search= [[-5.12, 5.12], [-5.12, 5.12]]

  gwo = GWO(
    fitness_function=sphere,
    space_search=space_search,
    max_iterations=100,
    agents_number=20)

  best_pos, best_fit = gwo.start()
  print("Best position:", best_pos)
  print("Best fitness :", best_fit)

  plotter = ObjectiveFunctionPlotter(sphere, lb=[-5.12, -5.12], ub=[5.12, 5.12])
  plotter.plot(points=[best_pos], best_point=best_pos, title="GWO on Sphere")
  plotter.plot_convergence(gwo.fitness_history)

  return best_pos, best_fit



if __name__ == "__main__":

  sheep_data = read_dataset()

  best_pos, best_fit = start_GWO()




