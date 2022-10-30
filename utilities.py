# Define context manager to measure execution time.

import time
import numpy as np

class codeTimer:
    """
    Context manager, measures and prints the execution time.
    """
    
    def __init__(self, name=None):
        self.name = "Executed '"  + name + "'. " if name else ""

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = (self.end - self.start)
        print('%s Elapsed time: %0.6fs' % (str(self.name), self.elapsed))
