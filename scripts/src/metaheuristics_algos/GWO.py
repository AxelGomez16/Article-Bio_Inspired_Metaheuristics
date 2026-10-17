import random
from typing import Callable, Optional, TypeAlias

Position: TypeAlias = list[float]
EvaluatedWolf: TypeAlias = tuple[int, Position, float]


class GWO:
  def __init__(
      self,
      fitness_function: Callable[[Position], float],
      space_search: list[list[float]],
      max_iterations: int = 100,
      agents_number: int = 20,
      mode: str = "minimize",
      tolerance: float = 1e-6
  ) -> None:
    """
    Initializes the Grey Wolf Optimizer (GWO).

    Parameters:
    - fitness_function: function to minimize. Receives a list (position) and returns a float.
    - space_search: list of lists with the [min, max] bounds for each dimension.
                    Example: [[-5, 5], [-5, 5]] for 2 dimensions.
    - max_iterations: maximum number of iterations (Default: 100).
    - agents_number: number of wolves (Default: 20).
    - mode: optimization mode, either "minimize" or "maximize" (Default: "minimize").
    - tolerance: tolerance for convergence (Default: 1e-6).
    """
    self.fitness_function = fitness_function
    self.space_search = space_search
    self.agents_number = agents_number
    self.max_iterations = max_iterations
    self.dimension = len(space_search)
    self.mode = mode
    self.tolerance = tolerance

    if self.mode == "minimize":
      self.is_maximizing = False
    elif self.mode == "maximize":
      self.is_maximizing = True
    else:
      raise ValueError("mode must be either 'minimize' or 'maximize'")
    if self.tolerance < 0:
      raise ValueError("tolerance must be non-negative")
    if self.agents_number < 3:
      raise ValueError("agents_number must be at least 3")
    if self.max_iterations < 1:
      raise ValueError("max_iterations must be at least 1")

    self.best_position: Optional[Position] = None
    self.best_fitness = self._initial_best_fitness()
    self.fitness_history: list[float] = []
    self.population_history: list[list[Position]] = []
    self.population_fitness_history: list[list[float]] = []

    self.plotter = None
    if self.dimension == 2:
      from scripts.src.visualizer.ObjectiveFunctionVisualizer import ObjectiveFunctionPlotter

      self.plotter = ObjectiveFunctionPlotter(
        fun_obj=self.fitness_function,
        space=self.space_search
      )


  def _random_position(self) -> Position:
    """Generates a random position inside the search space."""
    return [random.uniform(self.space_search[d][0],
                              self.space_search[d][1])
              for d in range(self.dimension)]

  def _evaluate(self, position: Position) -> float:
    """Evaluates the fitness of a given position."""
    return self.fitness_function(position)

  def _initial_best_fitness(self) -> float:
    """Returns the worst possible initial fitness for the selected mode."""
    return float("-inf") if self.is_maximizing else float("inf")

  def _is_better(self, candidate: float, current: float) -> bool:
    """Returns whether candidate improves the current fitness."""
    return candidate > current if self.is_maximizing else candidate < current

  def _is_inside_space_search(self, position: Position) -> Position:
    """Check the position so it stays inside the search space."""
    return [min(max(position[d], self.space_search[d][0]),
                  self.space_search[d][1])
              for d in range(self.dimension)]

  @staticmethod
  def _get_random_r() -> float:
    """Returns a random number uniformly distributed in [0, 1]."""
    return random.uniform(0, 1)

  def start(self) -> "GWO":
    """
    Runs the Grey Wolf Optimizer (GWO).

    Returns this optimizer instance with its results and histories populated.
    """

    self.best_position = None
    self.best_fitness = self._initial_best_fitness()
    self.fitness_history = []
    self.population_history = []
    self.population_fitness_history = []

    # Initialize the positions of the agents (Wolves)

    # Step 1: Initialize population of search agents (Wolves)
    wolves_positions: list[Position] = [
      self._random_position() for _ in range(self.agents_number)
    ]
    fitness: list[float] = [self._evaluate(w) for w in wolves_positions]
    self.population_history.append([position.copy() for position in wolves_positions])
    self.population_fitness_history.append(fitness.copy())

    for iteration in range(self.max_iterations):

      # Step 2: Initialize parameter "a^k"
      a_parameter = 2.0 - ((2.0 * iteration) / self.max_iterations)


      # Step 3: Identify wolves (Alpha, Beta, Delta, Omega) in the
      #         current population
      wolf_id, wolf_id_position, wolf_id_fitness_value = 0, 1, 2
      evaluated_wolves: list[EvaluatedWolf] = []
      for wolf in range(self.agents_number):
        evaluated_wolves.append((wolf, wolves_positions[wolf], fitness[wolf]))

      # Put the best three wolves first for either optimization mode.
      evaluated_wolves.sort(
        key=lambda wolf_: wolf_[2],
        reverse=self.is_maximizing
      )

      alpha_wolf = evaluated_wolves[0]
      beta_wolf = evaluated_wolves[1]
      delta_wolf = evaluated_wolves[2]
      omega_wolf = evaluated_wolves[3:]

      if self._is_better(alpha_wolf[wolf_id_fitness_value], self.best_fitness):
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
      best_wolf_id = (
        max(range(self.agents_number), key=lambda wolf_: fitness[wolf_])
        if self.is_maximizing
        else min(range(self.agents_number), key=lambda wolf_: fitness[wolf_])
      )
      previous_best_fitness = self.best_fitness
      if self._is_better(fitness[best_wolf_id], self.best_fitness):
        self.best_fitness = fitness[best_wolf_id]
        self.best_position = wolves_positions[best_wolf_id].copy()
      self.fitness_history.append(self.best_fitness)
      self.population_history.append([position.copy() for position in wolves_positions])
      self.population_fitness_history.append(fitness.copy())

      # TODO: Review the stopping criteria mmm
      # if abs(self.best_fitness - previous_best_fitness) <= self.tolerance:
      #   break

    # Step 5a: Return the best solution found
    if self.best_position is None:
      raise RuntimeError("GWO did not produce a best position")
    return self


  def visualize_movement(self, delay_interval: int = 250) -> None:
    """Show the movement of the wolf population over the objective surface."""
    if self.plotter is None:
      raise ValueError("Population surface visualization requires exactly 2 dimensions")

    self.plotter.animate_population(
      population_history=self.population_history,
      fitness_history=self.population_fitness_history,
      title="Grey Wolf Optimizer",
      interval=delay_interval,
    )

  def visualize_movement_html(self,):
    """Open the interactive html animation of the wolf population."""
    if self.plotter is None:
      raise ValueError("Population surface visualization requires exactly 2 dimensions")

    return self.plotter.animate_population_html(
      population_history=self.population_history,
      fitness_history=self.population_fitness_history,
      title="Grey Wolf Optimizer"
    )

  def visualize_charts(self):
    """Show the convergence and fitness charts of the wolf population."""
    from scripts.src.visualizer.ObjectiveFunctionVisualizer import ObjectiveFunctionPlotter

    if self.plotter is not None and self.best_position is not None:
      self.plotter.plot_best_solution(
        best_point=self.best_position,
        best_fitness=self.best_fitness,
        title="GWO Best Solution"
      )
    ObjectiveFunctionPlotter.plot_convergence(
      fitness_history=self.fitness_history,
      title="GWO Convergence Curve",
      log_scale=True
      )
