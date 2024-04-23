import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import torch


def animateBarycenterImages(atoms: torch.Tensor, barycenterSolver, height: int, width: int, n_frames=100, fps=30,
                            labels: str = None):
    if atoms.shape[1] != 2:
        raise ValueError("Must be only 2 atoms specified.")

    frames = torch.zeros((n_frames, atoms.shape[0], 1))
    alphas = torch.linspace(0, 1, n_frames)

    # first and last frame are the atoms
    frames[0] = atoms[:, 0]
    frames[-1] = atoms[:, 1]

    for i in range(1, n_frames - 1):
        interp_weights = torch.tensor([1 - alphas[i], alphas[i]]).view(-1, 1)
        b = barycenterSolver(atoms, interp_weights)
        frames[i] = b


def animateGrayImages(fig: plt.Figure, axes: np.ndarray, frames: [torch.Tensor], height: int, width: int, fps=30,
                      labels: str = None):
    """
    Animate a set of images

    :param frames: an (n_frames x (height x width) x n_images) tensor of data to animate
    :return:
    """

    n_rows = len(frames)

    # assert dimensions of objects line up as intended
    for i in range(len(frames) - 1):
        # same number of frames
        assert (frames[i].shape[0] == frames[i + 1].shape[0])

        # same number of columns (images) per row
        assert (frames[i].shape[2] == frames[i + 1].shape[2])

    # get relevant data
    n_frames = frames[0].shape[0]
    n_images = frames[0].shape[2]

    if labels is not None:
        for row in range(n_rows):
            for col in range(n_images):
                axes[row, col].set_title(labels[row][col])

    # get list of images to update
    ims = []
    for row in range(n_rows):
        ims.append([])
        for col in range(n_images):
            init_frame = 1 - (frames[row][0, :, col] / torch.max(frames[row][0, :, col]))
            init_frame /= torch.max(init_frame)
            ims[row].append(axes[row, col].imshow(init_frame.view(height, width), cmap="gray", vmin=0, vmax=1))

    def update(i):
        for row in range(n_rows):
            for col in range(n_images):
                new_frame = 1 - (frames[row][i, :, col] / torch.max(frames[row][i, :, col]))
                new_frame /= torch.max(new_frame)
                ims[row][col].set_array(new_frame.view(height, width))

    return FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
