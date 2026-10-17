import math

from scripts.src.metaheuristics_algos.GWO import GWO



def rastrigin(x):
  A = 10
  return A * len(x) + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)

def sphere(x: list[float]) -> float:
  return sum(xi ** 2 for xi in x)


def start_GWO() -> None:
  # Space search: 2 dimensions in [-5.12, 5.12]
  space_search = [[-5.12, 5.12], [-5.12, 5.12]]

  gwo = GWO(
    fitness_function=rastrigin,
    space_search=space_search,
    max_iterations=100,
    agents_number=20
  )

  gwo = gwo.start()


  best_pos = gwo.best_position
  best_fit = gwo.best_fitness
  print("Best position:", best_pos)
  print("Best fitness :", best_fit)

  gwo.visualize_movement(delay_interval=750)
  gwo.visualize_movement_html()
  gwo.visualize_charts()





if __name__ == "__main__":

  start_GWO()




