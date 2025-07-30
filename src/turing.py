import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, PngImagePlugin
from scipy.interpolate import interp1d
from scipy.ndimage import zoom


class TuringSimulator:
    def __init__(self, shape=(256, 256), **initial_params):
        self.shape = shape
        self.rgb = False
        self.reset()
        self.update_controls(initial_params)

    def reset(self):
        self.U = np.ones(self.shape)
        self.V = np.zeros(self.shape)
        self.seed()

    def seed(self, noise: float = 1.0):
        self.U += noise * (np.random.rand(*self.shape) - 0.5)
        self.V += noise * (np.random.rand(*self.shape) - 0.5)

    def update_controls(self, params):
        self.F = np.linspace(params["F1"], params["F2"], self.shape[1])[None, :].repeat(
            self.shape[0], axis=0
        )
        self.k = np.linspace(params["K1"], params["K2"], self.shape[1])[None, :].repeat(
            self.shape[0], axis=0
        )
        self.Du = np.linspace(params["Du1"], params["Du2"], self.shape[0])[
            :, None
        ].repeat(self.shape[1], axis=1)
        self.Dv = np.linspace(params["Dv1"], params["Dv2"], self.shape[0])[
            :, None
        ].repeat(self.shape[1], axis=1)

    def react(self):
        Lu, Lv = laplacian(self.U), laplacian(self.V)
        reaction = self.U * self.V * self.V
        self.U += self.Du * Lu - reaction + self.F * (1 - self.U)
        self.V += self.Dv * Lv + reaction - (self.F + self.k) * self.V

        # Check for invalid sim results.
        if np.isnan(self.U).any() or np.isnan(self.V).any():
            self.reset()

    def img_norm(self, Z):
        return (255 * (Z - Z.min()) / (Z.max() - Z.min())).astype(np.uint8)

    def step(self, steps: int = 1):
        for _ in range(steps):
            self.react()

        if self.rgb:
            u_img = self.img_norm(self.U)
            v_img = self.img_norm(self.V)
            return np.stack([u_img, np.zeros_like(u_img), v_img], axis=-1)
        else:
            return self.img_norm(self.V)


def initialize_grid(w: int, h: int, noise: float = 1.0) -> tuple[np.ndarray]:
    """Seed with random noise

    Args:
        w (int): Image pixel width
        h (int): image pixel height
        noise (float, optional): amplitude of initial noise. Defaults to 1.0.

    Returns:
        tuple[np.ndarray]: Initial map signals
    """
    U = np.ones((h, w))
    V = np.zeros((h, w))
    U += noise * (np.random.rand(h, w) - 0.5)
    V += noise * (np.random.rand(h, w) - 0.5)
    return U, V


def laplacian(Z: np.ndarray) -> np.ndarray:
    """Applies laplacian transform.

    Args:
        Z (np.ndarray): Input array

    Returns:
        np.ndarray: Laplacian of input aray
    """
    return (
        -Z
        + 0.2
        * (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))
        + 0.05
        * (
            np.roll(np.roll(Z, 1, 0), 1, 1)
            + np.roll(np.roll(Z, -1, 0), 1, 1)
            + np.roll(np.roll(Z, 1, 0), -1, 1)
            + np.roll(np.roll(Z, -1, 0), -1, 1)
        )
    )


def parameter_map(
    w: int,
    h: int,
    x_ctrl: list[float],
    p_vals: list[float],
    axis: str = "x",
) -> float | np.ndarray:
    """Generate variance map for a given control parameter used in diffusion calulations.

    Args:
        w (int): Image pixel width, x-axis.
        h (int): Image pixel height, y-axis.
        x_ctrl (list[float]): Parameter control points along fractional axis.
        p_vals (list[float]): Parameter values at control points.
        axis (str, optional): Axis along which the parameter varies ("x" or "y"). Defaults to "x".

    Returns:
        float | np.ndarray: Control parameter constant or map
    """
    # Validate
    if w <= 0 or h <= 0:
        raise ValueError("w and h must be integers greater than zero")
    if axis not in ["x", "y"]:
        raise ValueError("axis not in ['x','y']")
    if x_ctrl == [] or p_vals == []:
        raise ValueError("x_ctrl and p_vals must be specified")
    if len(x_ctrl) != len(p_vals):
        raise ValueError("x_ctrl and p_vals must have equal length")

    # Ensure numeric arrays
    x_array = np.array(x_ctrl)
    p_array = np.array(p_vals)
    if any(np.isnan(x_array)) or any(np.isnan(p_array)):
        raise ValueError("x_ctrl and p_vals must contain numbers")

    # Return constant if all p_vals are the same
    if np.ptp(p_array) == 0:
        return p_vals[0]

    # Create maps
    if axis == "x":
        x = np.linspace(0, 1, w)
        pvals = interp1d(x_array, p_array, kind="linear")(x)
        return np.tile(pvals, (h, 1))
    elif axis == "y":
        y = np.linspace(0, 1, h)
        pvals = interp1d(x_array, p_array, kind="linear")(y)
        return np.tile(pvals[:, np.newaxis], (1, w))


def turing_pattern(
    w: int = 512,
    h: int = 128,
    Du_ctrl: list[float] = [0.0, 1.0],
    Du_vals: list[float] = [0.7, 0.7],
    Du_axis: str = "y",
    Dv_ctrl: list[float] = [0.0, 1.0],
    Dv_vals: list[float] = [0.25, 0.25],
    Dv_axis: str = "y",
    F_ctrl: list[float] = [0.0, 1.0],
    F_vals: list[float] = [0.04, 0.08],
    F_axis: str = "x",
    k_ctrl: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    k_vals: list[float] = [0.056, 0.06, 0.0635, 0.0665, 0.07, 0.074],
    k_axis: str = "x",
    steps: int = 10000,
    upsample=2,
    **kwargs,
) -> np.ndarray:
    """Generate turing patterns.

    Args:
        w (int, optional): Image pixel width. Defaults to 512.
        h (int, optional): Image pixel height. Defaults to 128.
        Du_ctrl (list[float], optional): Component U diffusion axis control points. Defaults to [0.0, 1.0].
        Du_vals (list[float], optional): Component U diffusion values at control points. Defaults to [0.7, 0.7].
        Du_axis (str, optional): Axis along which to vary Du. Defaults to "y".
        Dv_ctrl (list[float], optional): Component V diffusion axis control points. Defaults to [0.0, 1.0].
        Dv_vals (list[float], optional): Component V diffusion values at control points. Defaults to [0.25, 0.25].
        Dv_axis (str, optional): Axis along which to vary Dv. Defaults to "y".
        F_ctrl (list[float], optional): Component U feed rate axis control points. Defaults to [0.0, 1.0].
        F_vals (list[float], optional): Component U feed rate values at control points. Defaults to [0.04, 0.08].
        F_axis (str, optional): Axis along which to vary F. Defaults to "x".
        k_ctrl (list[float], optional): Component V kill rate axis control points. Defaults to [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].
        k_vals (list[float], optional): Component V kill rate values at control points. Defaults to [0.056, 0.06, 0.0635, 0.0665, 0.07, 0.074].
        k_axis (str, optional): Axis along which to vary k. Defaults to "x".
        steps (int, optional): Iterations to run the simulation. Defaults to 10000.

    Returns:
        np.ndarray: Concentration map of component V
    """
    # Seed noise
    U, V = initialize_grid(w, h)

    # Generate control paramter maps
    Du = parameter_map(w, h, Du_ctrl, Du_vals, axis=Du_axis)
    Dv = parameter_map(w, h, Dv_ctrl, Dv_vals, axis=Dv_axis)
    F = parameter_map(w, h, F_ctrl, F_vals, axis=F_axis)
    k = parameter_map(w, h, k_ctrl, k_vals, axis=k_axis)

    # Plot setup
    if False:
        plt.ion()
        _, ax = plt.subplots()
        img = ax.imshow(V, cmap="gray", vmin=0, vmax=1)
        plt.title("Gray-Scott Turing Pattern")

    for i in range(steps):
        # Compute diffusion
        Lu, Lv = laplacian(U), laplacian(V)
        reaction = U * V * V
        U += Du * Lu - reaction + F * (1 - U)
        V += Dv * Lv + reaction - (F + k) * V

        # Perturb with noise in the beginning to encourage evolution
        if i % 1000 == 0 and i <= steps // 2:
            V += 0.5 * (np.random.rand(h, w) - 0.5)

        # Check status
        if i % 250 == 0:
            # Check model validity
            print(f"Step {i}/{steps}")
            ran = V.max() - V.min()
            if ran < 1e-5:
                raise ValueError(
                    f"Model collapsed by step {i}. Brightness range is to small: {ran}"
                )

            # Plot update
            if False:
                img.set_data(V)
                img.set_clim(vmin=V.min(), vmax=V.max())
                plt.pause(0.1)

    # Normalize V
    V_norm = (255 * (V - V.min()) / (V.max() - V.min())).astype(np.uint8)

    # Upsample image for ease of use in image editors later
    return zoom(V_norm, upsample, order=3)


def main():
    # Load parameters from JSON
    with open("turing_parameters.json", "r") as f:
        TuringParams = json.loads(f.read())

    # Generate pattern
    pattern = turing_pattern(**TuringParams)

    # Convert to Pillow image and create metadata
    img = Image.fromarray(pattern).convert("L")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("TuringParams", json.dumps(TuringParams, indent=2))

    # Save pattern with metadata
    now = datetime.strftime(datetime.now(), r"%Y%m%d%H%M%S")
    img.save(f"images/turing_pattern_{now}.png", pnginfo=metadata)


if __name__ == "__main__":
    main()
