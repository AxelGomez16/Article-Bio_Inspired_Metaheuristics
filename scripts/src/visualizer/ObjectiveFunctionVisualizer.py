import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Just for my pycharm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation




class ObjectiveFunctionPlotter:
  """
  Generic visualizer for 2D objective functions.

  Builds the 3D surface and 2D contour of any function f(x, y) and
  overlays arbitrary points (e.g. solutions from any metaheuristic).
  """
  def __init__(self,
               fun_obj,
               space: list[list[float]],
               n_samples: int = 50) -> None:
    """
    Parameters:
    - fun_obj:   objective function. Receives a list [x, y] and returns a float.
    - space:     bounds for each dimension, e.g. [[x_min, x_max], [y_min, y_max]].
    - n_samples: grid resolution for the surface (default: 50).
    """
    self.fun_obj = fun_obj
    self.space = space
    self.n_samples = n_samples
    if len(self.space) != 2:
      raise ValueError("space must contain bounds for exactly two dimensions")

    # Precompute the surface grid once
    self.x_axis, self.y_axis, self.z_axis = self._build_surface()

  # Internal: build the meshgrid + Z values
  def _build_surface(self):
    x_lin = np.linspace(self.space[0][0], self.space[0][1], self.n_samples)
    y_lin = np.linspace(self.space[1][0], self.space[1][1], self.n_samples)

    z_axis = np.zeros((self.n_samples, self.n_samples))
    for i in range(self.n_samples):
      for j in range(self.n_samples):
        z_axis[j, i] = self.fun_obj([x_lin[i], y_lin[j]])

    # meshgrid: rows -> y, cols -> x
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    return x_grid, y_grid, z_axis

  def plot_best_solution(
      self,
      best_point: list[float] | np.ndarray,
      best_fitness: float,
      title: str = "Objective Function"
  ) -> None:
    """
    Parameters:
    - best_point: [x, y] coordinates of the best solution.
    - best_fitness: objective value for the best solution.
    - title:      figure title.
    """
    bp = np.asarray(best_point, dtype=float)
    if bp.shape != (2, ):
      raise ValueError("best_point must have shape (2,)")
    fig = plt.figure(figsize=(12, 6))

    # -------- 3D surface --------
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(
      self.x_axis,
      self.y_axis,
      self.z_axis,
      cmap="viridis",
      alpha=0.8
    )
    ax1.scatter(
      bp[0],
      bp[1],
      best_fitness,
      c="gold",
      marker="*",
      s=200,
      edgecolor="black",
      label="Best solution"
    )

    ax1.set_xlabel("x_1")
    ax1.set_ylabel("x_2")
    ax1.set_zlabel("f(x)")
    ax1.set_title("3D Surface Plot")
    ax1.legend()

    # -------- 2D contour --------
    ax2 = fig.add_subplot(122)
    ax2.contour(
      self.x_axis,
      self.y_axis,
      self.z_axis,
      levels=20,
      cmap="viridis",
      zorder=1
    )
    ax2.scatter(
      bp[0],
      bp[1],
      c="gold",
      marker="*",
      s=200,
      edgecolor="black",
      zorder=10,
      label="Best solution"
    )

    ax2.set_xlabel("x_1")
    ax2.set_ylabel("x_2")
    ax2.set_title("2D Contour Plot")
    ax2.legend()
    ax2.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

  def animate_population(
      self,
      population_history: list[list[list[float]]],
      fitness_history: list[list[float]] | None = None,
      title: str = "Population Movement",
      interval: int = 250
  ) -> FuncAnimation:
    """Animate a 2D population history over the objective surface."""
    if not population_history:
      raise ValueError(
        "population_history must contain at least one population")

    populations = [
      np.asarray(population, dtype=float) for population in population_history
    ]
    agent_count = populations[0].shape[0]
    if agent_count == 0 or any(population.shape != (agent_count, 2)
                                for population in populations):
      raise ValueError(
        "each population must have the same nonzero shape (N, 2)")

    if fitness_history is None:
      fitness_values = [
        np.asarray([self.fun_obj(point) for point in population])
        for population in populations
      ]
    else:
      fitness_values = [
        np.asarray(values, dtype=float) for values in fitness_history
      ]
      if (len(fitness_values) != len(populations)
          or any(values.shape != (agent_count, )
                  for values in fitness_values)):
        raise ValueError("fitness_history must match the population history")

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, agent_count))
    figure = plt.figure(figsize=(12, 6))
    surface_axis = figure.add_subplot(121, projection="3d")
    surface_axis.plot_surface(
      self.x_axis,
      self.y_axis,
      self.z_axis,
      cmap="viridis",
      alpha=0.65
    )
    initial_positions = populations[0]
    initial_fitness = fitness_values[0]
    initial_best_index = int(np.argmin(initial_fitness))

    surface_points = surface_axis.scatter(
      initial_positions[:, 0],
      initial_positions[:, 1],
      initial_fitness,
      c=colors,
      s=35
    )
    surface_best = surface_axis.scatter(
      [initial_positions[initial_best_index, 0]],
      [initial_positions[initial_best_index, 1]],
      [initial_fitness[initial_best_index]],
      c="gold",
      marker="*",
      s=180,
      edgecolor="black")
    surface_axis.set_xlabel("x_1")
    surface_axis.set_ylabel("x_2")
    surface_axis.set_zlabel("f(x)")
    surface_axis.set_title("3D Surface Plot")

    contour_axis = figure.add_subplot(122)
    contour_axis.contour(
      self.x_axis,
      self.y_axis,
      self.z_axis,
      levels=20,
      cmap="viridis",
      zorder=1
    )
    contour_points = contour_axis.scatter(
      initial_positions[:, 0],
      initial_positions[:, 1],
      c=colors,
      s=35,
      zorder=2
    )
    contour_best = contour_axis.scatter(
      [initial_positions[initial_best_index, 0]],
      [initial_positions[initial_best_index, 1]],
      c="gold",
      marker="*",
      s=180,
      edgecolor="black",
      zorder=3
    )
    contour_axis.set_xlabel("x_1")
    contour_axis.set_ylabel("x_2")
    contour_axis.set_title("2D Contour Plot")
    contour_axis.set_aspect("equal", adjustable="box")

    figure.suptitle(title, fontsize=14, y=0.99)
    subtitle = figure.text(0.5, 0.95, "", ha="center", va="top", fontsize=10)

    def update(frame: int):
      positions = populations[frame]
      scores = fitness_values[frame]
      best_index = int(np.argmin(scores))
      surface_points._offsets3d = (positions[:, 0], positions[:, 1], scores)
      surface_best._offsets3d = (
        [positions[best_index, 0]], [positions[best_index, 1]], [scores[best_index]])
      contour_points.set_offsets(positions)
      contour_best.set_offsets(positions[[best_index]])

      subtitle.set_text(f"Iteration {frame}/{len(populations) - 1}"
                        f" | Best fitness: {scores[best_index]:.6g}")
      return surface_points, surface_best, contour_points, contour_best

    animation = FuncAnimation(figure,
                              update,
                              frames=len(populations),
                              interval=interval,
                              repeat=False,
                              blit=False)
    figure.tight_layout(rect=(0, 0.02, 1, 0.90))
    plt.show()

    return animation

  def animate_population_html(
    self,
    population_history: list[list[list[float]]],
    fitness_history: list[list[float]] | None = None,
    title: str = "Population Movement",
  ) -> "go.Figure":

    if not population_history:
      raise ValueError(
        "population_history must contain at least one population")

    populations = [
      np.asarray(population, dtype=float) for population in population_history
    ]
    agent_count = populations[0].shape[0]
    if agent_count == 0 or any(population.shape != (agent_count, 2)
                               for population in populations):
      raise ValueError(
        "each population must have the same nonzero shape (N, 2)")

    if fitness_history is None:
      fitness_values = [
        np.asarray([self.fun_obj(point) for point in population])
        for population in populations
      ]
    else:
      fitness_values = [
        np.asarray(values, dtype=float) for values in fitness_history
      ]
      if (len(fitness_values) != len(populations)
          or any(values.shape != (agent_count, )
                 for values in fitness_values)):
        raise ValueError("fitness_history must match the population history")

    colors = [
      "#e45756", "#4c78a8", "#f2cf5b", "#72b7b2", "#b279a2", "#ff9da6",
      "#9d755d", "#bab0ac", "#59a14f", "#edc949"
    ]
    agent_colors = [
      colors[index % len(colors)] for index in range(agent_count)
    ]
    x_values = self.x_axis[0, :]
    y_values = self.y_axis[:, 0]
    dynamic_trace_indices = list(range(2, 6))

    def frame_traces(frame_index: int):
      positions = populations[frame_index]
      scores = fitness_values[frame_index]
      best_index = int(np.argmin(scores))
      traces = [
        go.Scatter3d(
          x=positions[:, 0],
          y=positions[:, 1],
          z=scores,
          mode="markers",
          marker={
            "color": agent_colors,
            "size": 10
          },
          text=[f"Wolf {index}" for index in range(agent_count)],
          name="Wolves 3D",
          hovertemplate=("%{text}<br>x: %{x:.4g}<br>y: %{y:.4g}"
                                    "<br>fitness: %{z:.4g}<extra></extra>")
        ),
        go.Scatter3d(
          x=[positions[best_index, 0]],
          y=[positions[best_index, 1]],
          z=[scores[best_index]],
          mode="markers",
          marker={
            "color": "gold",
            "size": 15,
            "symbol": "diamond",
            "line": {
              "color": "black",
              "width": 2
            }
          },
          name="Best Solution 3D",
          hovertemplate="Best fitness: %{z:.6g}<extra></extra>"
        ),
        go.Scatter(
          x=positions[:, 0],
          y=positions[:, 1],
          mode="markers",
          marker={
            "color": agent_colors,
            "size": 10
          },
          text=[f"Wolf {index}" for index in range(agent_count)],
          name="Wolves 2D",
          hovertemplate=("%{text}<br>x: %{x:.4g}<br>y: %{y:.4g}"
                                  "<extra></extra>")
        ),
        go.Scatter(
          x=[positions[best_index, 0]],
          y=[positions[best_index, 1]],
          mode="markers",
          marker={
            "color": "gold",
            "size": 20,
            "symbol": "star",
            "line": {
              "color": "black",
              "width": 2
            }
          },
          name="Best Solution 2D",
          hovertemplate="Best wolf<extra></extra>"
        ),
      ]
      return traces

    figure = make_subplots(
      rows=1,
      cols=2,
      specs=[[{
        "type": "scene"
      }, {
        "type": "xy"
      }]],
      subplot_titles=("Objective surface", "Objective contour"),
      horizontal_spacing=0.08,
    )
    figure.add_trace(go.Surface(
      x=self.x_axis,
      y=self.y_axis,
      z=self.z_axis,
      colorscale="Viridis",
      opacity=0.75,
      showscale=False,
      hovertemplate=
      "x: %{x:.4g}<br>y: %{y:.4g}<br>f(x): %{z:.4g}<extra></extra>",
      name="Objective function",
    ),
                     row=1,
                     col=1)
    figure.add_trace(go.Contour(
      x=x_values,
      y=y_values,
      z=self.z_axis,
      colorscale="Viridis",
      ncontours=20,
      showscale=False,
      contours={"coloring": "lines"},
      hoverinfo="skip",
      name="Contours",
    ),
                     row=1,
                     col=2)

    initial_traces = frame_traces(0)
    for trace in initial_traces[:2]:
      figure.add_trace(trace, row=1, col=1)
    for trace in initial_traces[2:4]:
      figure.add_trace(trace, row=1, col=2)
    frames = []
    for frame_index in range(len(populations)):
      scores = fitness_values[frame_index]
      best_fitness = float(np.min(scores))
      frames.append(
        go.Frame(
          name=str(frame_index),
          data=frame_traces(frame_index),
          traces=dynamic_trace_indices,
          layout={
            "title":
            f"{title} | Iteration {frame_index}"
            f"/{len(populations) - 1} | Best fitness: {best_fitness:.6g}"
          },
        ))

    slider_steps = [{
      "method":
      "animate",
      "label":
      str(frame_index),
      "args": [[str(frame_index)], {
        "mode": "immediate",
        "frame": {
          "duration": 0,
          "redraw": True
        },
        "transition": {
          "duration": 0
        }
      }]
    } for frame_index in range(len(populations))]
    initial_best_fitness = float(np.min(fitness_values[0]))
    figure.frames = frames
    figure.update_layout(
      title=f"{title} | Iteration 0/{len(populations) - 1}"
      f" | Best fitness: {initial_best_fitness:.6g}",
      scene={
        "xaxis_title": "x₁",
        "yaxis_title": "x₂",
        "zaxis_title": "f(x)"
      },
      xaxis={"title": "x₁"},
      yaxis={
        "title": "x₂",
        "scaleanchor": "x",
        "scaleratio": 1
      },
      sliders=[{
        "active": 0,
        "x": 0.0,
        "y": -0.18,
        "len": 1.0,
        "currentvalue": {
          "prefix": "Iteration: "
        },
        "steps": slider_steps
      }],
      margin={"b": 100},
    )
    figure.show()
    return figure

  @staticmethod
  def plot_convergence(
    fitness_history: list[float],
    title: str = "Convergence Curve",
    log_scale: bool = True
  ) -> None:
    """Plot any list of fitness values over iterations."""
    plt.figure(figsize=(8, 5))
    if log_scale:
      plt.semilogy(fitness_history, linewidth=2)
      plt.ylabel("Fitness (log scale)")
    else:
      plt.plot(fitness_history, linewidth=2)
      plt.ylabel("Fitness")
    plt.xlabel("Iterations")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
