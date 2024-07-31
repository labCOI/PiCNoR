import logging
import numpy as np
import traceback

class CustomLogger(logging.Logger):
    def __init__(self, name, log_file, level=logging.INFO):
        super().__init__(name, level)
        self.log_file = log_file
        self.setup()

    def setup(self):

        file_handler = logging.FileHandler(self.log_file,  mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)

    def message(self, msg):
        """
        Log a message.
        
        Args:
            msg (str): The message to log.
        """
        self.info(msg)

    def log_error(self, msg, exc_info=True):
        """
        Log an error message with optional exception information.
        
        Args:
            msg (str): The error message to log.
            exc_info (bool): If True, include exception information in the log.
        """
        if exc_info:
            self.error(f"{msg}\nException details: {traceback.format_exc()}")
        else:
            self.error(msg)

    def log_transform_info(self, index, M, inlierIndex):
        np.set_printoptions(precision=3, suppress=True, linewidth=100)
        
        tx, ty = M[0, 2], M[1, 2]
        rotation = np.degrees((np.arccos(M[0, 0])))
        if index != -1:
            log_message = (
                f"\nCluster {index}:\n"
                f"Global Transform Parameters: (tx={tx:.5f}, ty={ty:.5f}, rotation={rotation:.5f} degrees)\n"
                f"Inliers: {np.sum(inlierIndex)}"
            )
        else:
            log_message = (
                f"Global Transform Parameters: (tx={tx:.5f}, ty={ty:.5f}, rotation={rotation:.5f} degrees)\n"
                f"Inliers: {np.sum(inlierIndex)}"
            )
        self.info(log_message)

    def log_section_divider(self):
        self.info("-" * 40)

class CustomFullLogger(logging.Logger):
    def __init__(self, name, log_file, level=logging.INFO):
        super().__init__(name, level)
        self.log_file = log_file
        self.setup()

    def setup(self):

        file_handler = logging.FileHandler(self.log_file,  mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        self.addHandler(console_handler)

    def message(self, msg):
        """
        Log a message.
        
        Args:
            msg (str): The message to log.
        """
        self.info(msg)
    
    def log_error(self, msg, exc_info=True):
        """
        Log an error message with optional exception information.
        
        Args:
            msg (str): The error message to log.
            exc_info (bool): If True, include exception information in the log.
        """
        if exc_info:
            self.error(f"{msg}\nException details: {traceback.format_exc()}")
        else:
            self.error(msg)

    def log_transform_info(self, index, M, inlierIndex):
        np.set_printoptions(precision=3, suppress=True, linewidth=100)
        
        M_formatted = np.array2string(M, formatter={'float_kind':lambda x: f"{x:10.5f}"})
        tx, ty = M[0, 2], M[1, 2]
        rotation = np.degrees((np.arccos(M[0, 0])))
        if index != -1:
            log_message = (
                f"Cluster {index}:\n"
                f"Transform Matrix:\n{M_formatted}\n"
                f"Transform Parameters: (tx={tx:.5f}, ty={ty:.5f}, rotation={rotation:.5f} degrees)\n"
                f"Inliers: {np.sum(inlierIndex)}"
            )
        else:
            log_message = (
                f"Global Transform Matrix:\n{M_formatted}\n"
                f"Transform Parameters: (tx={tx:.5f}, ty={ty:.5f}, rotation={rotation:.5f} degrees)\n"
                f"Inliers: {np.sum(inlierIndex)}"
            )
        self.info(log_message)
 
    def log_section_divider(self):
        self.info("-" * 40)
