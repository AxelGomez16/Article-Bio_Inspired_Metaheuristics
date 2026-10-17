from pathlib import Path

import numpy as np
import pandas as pd

from scripts.src.metaheuristics_algos.GWO import GWO

KINECT_FEATURES = {
  "HWK" : "height_at_withers_kinect_HWK",
  "RHK" : "rump_height_kinect_RHK",
  "BLK" : "body_length_kinect_BLK",
  "BDLK" : "body_diagonal_length_kinect_BDLK",
  "BTLK" : "body_total_length_kinect_BTLK",
  "GSCK" : "thoracic_girth_semi_circumference_kinect_GSCK",
  "ASCK" : "abdominal_semi_circumference_kinect_ASCK",
}


class SheepWeightEstimator:
  def __init__(self):
    self.data = self.read_dataset()

  def read_dataset(self) -> pd.DataFrame:
    dataset_path = Path(__file__).resolve().parent / "dataset" / "data_sheep_body_weight.csv"
    return pd.read_csv(dataset_path)

  def fitness_function(
      self,
      parameters: list[float]
    ) -> float:
    biometric_values = self.data[list(KINECT_FEATURES.values())]
    real_weights = self.data["body_weight"]
    
    estimated_weights = self.calculate_bwe(
      parameters,
      biometric_values
    )
    return self.calculate_rmse(estimated_weights, real_weights)


  @staticmethod
  def is_columns_exist(data: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in data.columns for column in columns)


  @staticmethod
  def calculate_rmse(
      estimated_weights: pd.Series,
      real_weights: pd.Series
    ) -> float:
    """Calculate RMSE between estimated and measured body weights."""
    
    if len(estimated_weights) != len(real_weights) or len(estimated_weights) == 0:
      raise ValueError("estimated_weights and actual_weights must be nonempty arrays of equal shape")

    sum_bwe = 0
    for bwe, bw in zip(estimated_weights, real_weights):
      sum_bwe += (bwe - bw) ** 2
    rmse_value = np.sqrt(sum_bwe / len(real_weights))

    return rmse_value

  def calculate_bwe(
      self,
      parameters: list[float],
      biometric_values: pd.DataFrame
    ) -> pd.Series:
    """Estimate body weight using seven coefficients and seven exponents."""

    if len(parameters) != 14:
      raise ValueError("parameters must contain 7 coefficients followed by 7 exponents")
    if not self.is_columns_exist(biometric_values, list(KINECT_FEATURES.values())):
      raise ValueError("biometric_values must have shape (number_of_sheep, 7)")

    HWK = biometric_values[KINECT_FEATURES["HWK"]]
    RHK = biometric_values[KINECT_FEATURES["RHK"]]
    BLK = biometric_values[KINECT_FEATURES["BLK"]]
    BDLK = biometric_values[KINECT_FEATURES["BDLK"]]
    BTLK = biometric_values[KINECT_FEATURES["BTLK"]]
    GSCK = biometric_values[KINECT_FEATURES["GSCK"]]
    ASCK = biometric_values[KINECT_FEATURES["ASCK"]]

    bwe_values_ = (
      parameters[0] * np.power(HWK, parameters[7]) +
      parameters[1] * np.power(RHK, parameters[8]) +
      parameters[2] * np.power(BLK, parameters[9]) +
      parameters[3] * np.power(BDLK, parameters[10]) +
      parameters[4] * np.power(BTLK, parameters[11]) +
      parameters[5] * np.power(GSCK, parameters[12]) +
      parameters[6] * np.power(ASCK, parameters[13])
    )

    return pd.Series(bwe_values_)



  def run_sheep_gwo(self) -> GWO:
    # coefficients a1-a7 in [0, 1], exponents a8-a14 in [0, 5].
    space_search = [[0.0, 1.0] for _ in range(7)] + [[0.0, 5.0] for _ in range(7)]

    gwo = GWO(
      fitness_function=self.fitness_function,
      space_search=space_search,
      max_iterations=100,
      agents_number=20,
    ).start()

    best_parameters = np.asarray(gwo.best_position, dtype=float)
    # print(f"Measurements used: {len(self.data)}")
    print("Best coefficients (a1-a7):", best_parameters[:7].tolist())
    print("Best exponents (a8-a14):", best_parameters[7:].tolist())
    print("Best fitness value:", gwo.best_fitness)

    # gwo.visualize_charts()
    return gwo


def plot_best_fitness(results: list[float]):
  """Plot best RMSE values across optimization runs."""
  import matplotlib.pyplot as plt

  if not results:
    raise ValueError("results must contain at least one fitness value")

  run_numbers = np.arange(1, len(results) + 1)
  figure, axis = plt.subplots(figsize=(9, 5))
  axis.scatter(run_numbers, results, color="#4c78a8", s=36)
  axis.set_title("Best fitness value across runs")
  axis.set_xlabel("Run")
  axis.set_ylabel("RMSE (body-weight)")
  tick_positions = np.arange(1, len(results) + 1, 5)
  if tick_positions[-1] != len(results):
    tick_positions = np.append(tick_positions, len(results))
  axis.set_xticks(tick_positions)
  axis.grid(True, alpha=0.3)
  figure.tight_layout()
  plt.show()
  return figure


def plot_best_parameters(results: list[list[float]]):
  """Plot each run's best parameter position as a line with markers."""
  import matplotlib.pyplot as plt

  if not results:
    raise ValueError("results must contain at least one optimization run")
  if any(len(parameters) != 14 for parameters in results):
    raise ValueError("each best parameter position must contain 14 values")

  parameter_labels = [f"a{index + 1}" for index in range(14)]
  colors = plt.get_cmap("tab10").colors
  figure, axis = plt.subplots(figsize=(11, 5))

  for index, best_position in enumerate(results):
    axis.plot(
      parameter_labels,
      best_position,
      marker="o",
      linewidth=1.5,
      color=colors[index % len(colors)],
      label=f"Run {index + 1}",
    )

  axis.set_title("Best parameter position per run")
  axis.set_xlabel("Model parameter (a1-a7: coefficients; a8-a14: exponents)")
  axis.set_ylabel("Parameter value")
  axis.set_xticks(range(len(parameter_labels)), parameter_labels)
  axis.grid(True, alpha=0.3)
  axis.legend()
  figure.tight_layout()
  plt.show()
  return figure


if __name__ == "__main__":
  estimator = SheepWeightEstimator()

  results_best_fitness = []
  results_best_position = [] 
  
  for _ in range(5):
    result = estimator.run_sheep_gwo()
    results_best_fitness.append(result.best_fitness)
    results_best_position.append(result.best_position)

  plot_best_fitness(results_best_fitness)
  plot_best_parameters(results_best_position)


