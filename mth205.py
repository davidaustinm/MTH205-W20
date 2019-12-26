import numpy as np
import numpy.linalg as LA

class matrix:
    def __init__(self, rows, cols, entries):
        self.entries = np.reshape(entries, (rows, cols))
        self.rows = rows
        self.cols = cols
    def eigenvalues(self):
        return LA.eig(self.entries)[0]

                                  
