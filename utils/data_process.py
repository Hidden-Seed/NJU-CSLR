import numpy as np
import scipy.io as scio

from utils.logger import Logger


def txt2mat(txt_file_path, mat_path, logger: Logger):
    """
    Convert a .txt file to a .mat file.

    Args:
        txt_path (str): Path to the input .txt file.
        mat_path (str): Path to the output .mat file.
    """

    # Read txt as flat array
    data = np.loadtxt(txt_file_path).reshape(-1, 138)

    # Save to .mat
    scio.savemat(mat_path, {'data': data})
    logger.info(f"Saved to: {mat_path}")
