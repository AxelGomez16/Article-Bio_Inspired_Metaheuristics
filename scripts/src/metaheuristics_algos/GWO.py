import random
from typing import Callable, Optional, TypeAlias


Position: TypeAlias = list[float]
EvaluatedWolf: TypeAlias = tuple[int, Position, float]


class GWO:
  def __init__(self,
              fitness_function: Callable[[Position], float],
              space_search: list[list[float]],
              max_iterations: int = 100,
              agents_number: int = 20
              ) -> None :
    """
    Initializes the Grey Wolf Optimizer (GWO).

    Parameters:
    - fitness_function: function to minimize. Receives a list (position) and returns a float.
    - space_search: list of lists with the [min, max] bounds for each dimension.
                    Example: [[-5, 5], [-5, 5]] for 2 dimensions.
    - max_iterations: maximum number of iterations (Default: 100).
    - agents_number: number of wolves (Default: 20).
    """
    self.fitness_function = fitness_function
    self.space_search = space_search
    self.agents_number = agents_number
    self.max_iterations = max_iterations
    self.dimension = len(space_search)

    if self.agents_number < 3:
      raise ValueError("agents_number must be at least 3")
    if self.max_iterations < 1:
      raise ValueError("max_iterations must be at least 1")

    self.best_position: Optional[Position] = None
    self.best_fitness = float('inf')
    self.fitness_history: list[float] = []


  def _random_position(self) -> Position:
    """Generates a random position inside the search space."""
    return [random.uniform(self.space_search[d][0],
                              self.space_search[d][1])
              for d in range(self.dimension)]

  def _evaluate(self, position: Position) -> float:
    """Evaluates the fitness of a given position."""
    return self.fitness_function(position)

  def _is_inside_space_search(self, position: Position) -> Position:
    """Check the position so it stays inside the search space."""
    return [min(max(position[d], self.space_search[d][0]),
                  self.space_search[d][1])
              for d in range(self.dimension)]

  @staticmethod
  def _get_random_r() -> float:
    """Returns a random number uniformly distributed in [0, 1]."""
    return random.uniform(0, 1)

  def start(self) -> tuple[Position, float]:
    """
    Runs the Grey Wolf Optimizer (GWO).

    Returns:
    - best_position: position of the best wolf found.
    - best_fitness: fitness value of that wolf.
    """

    # Initialize the positions of the agents (Wolves)

    # Step 1: Initialize population of search agents (Wolves)
    wolves_positions: list[Position] = [
      self._random_position() for _ in range(self.agents_number)
    ]
    fitness: list[float] = [self._evaluate(w) for w in wolves_positions]

    for iteration in range(self.max_iterations):

      # Step 2: Initialize parameter "a^k"
      a_parameter = 2.0 - ((2.0 * iteration) / self.max_iterations)


      # Step 3: Identify wolves (Alpha, Beta, Delta, Omega) in the
      #         current population
      wolf_id, wolf_id_position, wolf_id_fitness_value = 0, 1, 2
      evaluated_wolves: list[EvaluatedWolf] = []
      for wolf in range(self.agents_number):
        evaluated_wolves.append((wolf, wolves_positions[wolf], fitness[wolf]))

      # Order from least to greatest fitness value (Minimization)
      evaluated_wolves.sort(key=lambda wolf_: wolf_[2])

      alpha_wolf = evaluated_wolves[0]
      beta_wolf = evaluated_wolves[1]
      delta_wolf = evaluated_wolves[2]
      omega_wolf = evaluated_wolves[3:]

      if alpha_wolf[wolf_id_fitness_value] < self.best_fitness:
        self.best_fitness = alpha_wolf[wolf_id_fitness_value]
        self.best_position = alpha_wolf[wolf_id_position].copy()

      # Step 4: Move wolves
      for moved_wolf in range(self.agents_number):
        new_position = [0.0] * self.dimension

        for dimension in range(self.dimension):
          current_moved_wolf_position = evaluated_wolves[moved_wolf][wolf_id_position]

          # Alpha wolf
          r1_alpha, r2_alpha = self._get_random_r(), self._get_random_r()
          A_alpha = (2.0 * a_parameter * r1_alpha) - a_parameter
          C_alpha = (2.0 * r2_alpha)
          position_alpha = alpha_wolf[wolf_id_position][dimension] - (A_alpha * abs(C_alpha * alpha_wolf[wolf_id_position][dimension] - current_moved_wolf_position[dimension]))

          # Beta wolf
          r1_beta, r2_beta = self._get_random_r(), self._get_random_r()
          A_beta = (2.0 * a_parameter * r1_beta) - a_parameter
          C_beta = (2.0 * r2_beta)
          position_beta = beta_wolf[wolf_id_position][dimension] - (A_beta * abs(C_beta * beta_wolf[wolf_id_position][dimension] - current_moved_wolf_position[dimension]))

          # Delta wolf
          r1_delta, r2_delta = self._get_random_r(), self._get_random_r()
          A_delta = (2.0 * a_parameter * r1_delta) - a_parameter
          C_delta = (2.0 * r2_delta)
          position_delta = delta_wolf[wolf_id_position][dimension] - (A_delta * abs(C_delta * delta_wolf[wolf_id_position][dimension] - current_moved_wolf_position[dimension]))

          new_position[dimension] = (position_alpha + position_beta + position_delta) / 3.0

        # evaluated_wolves is sorted, so update the original wolf by its ID.
        original_wolf_id = evaluated_wolves[moved_wolf][wolf_id]
        wolves_positions[original_wolf_id] = self._is_inside_space_search(new_position)
        fitness[original_wolf_id] = self._evaluate(wolves_positions[original_wolf_id])

      # Step 5: Check the stopping criterion
      # Step 5b: Evaluate the condition of stop and save the best solution till now
      best_wolf_id = min(range(self.agents_number), key=lambda wolf_: fitness[wolf_])
      if fitness[best_wolf_id] < self.best_fitness:
        self.best_fitness = fitness[best_wolf_id]
        self.best_position = wolves_positions[best_wolf_id].copy()
      self.fitness_history.append(self.best_fitness)

    # Step 5a: Return the best solution found
    if self.best_position is None:
      raise RuntimeError("GWO did not produce a best position")
    return self.best_position, self.best_fitness

