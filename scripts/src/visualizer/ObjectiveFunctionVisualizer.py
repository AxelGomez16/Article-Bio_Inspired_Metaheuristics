import numpy as np
import matplotlib.pyplot as plt


class ObjectiveFunctionPlotter:
    """
    Generic visualizer for 2D objective functions.

    Builds the 3D surface and 2D contour of any function f(x, y) and
    overlays arbitrary points (e.g. solutions from any metaheuristic).
    """

    def __init__(self,
                 fun_obj,
                 lb: list[float],
                 ub: list[float],
                 n_samples: int = 50) -> None:
        """
        Parameters:
        - fun_obj:   objective function. Receives a list [x, y] and returns a float.
        - lb:        lower bounds [x_min, y_min].
        - ub:        upper bounds [x_max, y_max].
        - n_samples: grid resolution for the surface (default: 50).
        """
        self.fun_obj = fun_obj
        self.lb = lb
        self.ub = ub
        self.n_samples = n_samples

        # Precompute the surface grid once
        self.x_axis, self.y_axis, self.z_axis = self._build_surface()

    # ------------------------------------------------------------------
    # Internal: build the meshgrid + Z values
    # ------------------------------------------------------------------
    def _build_surface(self):
        x_lin = np.linspace(self.lb[0], self.ub[0], self.n_samples)
        y_lin = np.linspace(self.lb[1], self.ub[1], self.n_samples)

        z_axis = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                z_axis[i, j] = self.fun_obj([x_lin[i], y_lin[j]])

        # meshgrid: rows -> y, cols -> x
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)
        return x_grid, y_grid, z_axis


    def plot(self,
             points: list[list[float]] | np.ndarray,
             best_point: list[float] | None = None,
             title: str = "Objective Function") -> None:
        """
        Parameters:
        - points:     list/array of shape (N, 2) with any set of 2D points.
        - best_point: optional [x, y] highlighted as the best solution.
        - title:      figure title.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        # Fitness of every point (Z coordinate for the 3D scatter)
        pts_fitness = np.array([self.fun_obj(p) for p in pts])

        fig = plt.figure(figsize=(12, 6))

        # -------- 3D surface --------
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(self.x_axis, self.y_axis, self.z_axis,
                         cmap="viridis", alpha=0.8)
        ax1.scatter(pts[:, 0], pts[:, 1], pts_fitness,
                    c="red", marker="o", s=50, label="Points")

        if best_point is not None:
            bp = np.asarray(best_point, dtype=float)
            ax1.scatter(bp[0], bp[1], self.fun_obj(bp),
                        c="gold", marker="*", s=200,
                        edgecolor="black", label="Best")

        ax1.set_xlabel("x₁")
        ax1.set_ylabel("x₂")
        ax1.set_zlabel("f(x)")
        ax1.set_title("Surface")
        ax1.legend()

        # -------- 2D contour --------
        ax2 = fig.add_subplot(122)
        ax2.contour(self.x_axis, self.y_axis, self.z_axis,
                    levels=20, cmap="viridis")
        ax2.scatter(pts[:, 0], pts[:, 1],
                    c="red", marker="o", s=50, label="Points")

        if best_point is not None:
            bp = np.asarray(best_point, dtype=float)
            ax2.scatter(bp[0], bp[1],
                        c="gold", marker="*", s=200,
                        edgecolor="black", label="Best")

        ax2.set_xlabel("x₁")
        ax2.set_ylabel("x₂")
        ax2.set_title("Contour")
        ax2.legend()
        ax2.set_aspect("equal", adjustable="box")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_convergence(history: list[float],
                         title: str = "Convergence Curve",
                         log_scale: bool = True) -> None:
        """Plot any list of fitness values over iterations."""
        plt.figure(figsize=(8, 5))
        if log_scale:
            plt.semilogy(history, linewidth=2)
            plt.ylabel("Fitness (log scale)")
        else:
            plt.plot(history, linewidth=2)
            plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()