import numpy as np
import numpy.linalg as LA

class matrix:
    def __init__(self, *args):
        if len(args) == 3:
            rows = args[0]
            cols = args[1]
            entries = args[2]
            self.entries = np.reshape(entries, (rows, cols))
            self.rows = rows
            self.cols = cols
        if len(args) == 1:
            self.entries = args[0]
            shape = np.shape(self.entries)
            self.rows = shape[0]
            self.cols = shape[1]
    def eigenvalues(self):
        return LA.eig(self.entries)[0]
    def right_eigenmatrix(self):
        values, vectors = LA.eig(self.entries)
        return (values, matrix(vectors))
    def __add__(self, B):
        if isinstance(B, matrix):
            return matrix(self.entries + B.entries)
    def __sub__(self, B):
        if isinstance(B, matrix):
            return matrix(self.entries - B.entries)
    def __mul__(self, B):
        if isinstance(B, matrix): 
            return matrix(self.entries.dot(B.entries))
        if isinstance(B, vector):
            return vector(self.entries.dot(B.entries.T))
    def __rmul__(self, s):
        return matrix(s*self.entries)
    def __str__(self):
        return str(self.entries)

class vector:
    def __init__(self, entries):
        self.entries = np.array(entries)
    def __str__(self):
        return str(self.entries)
    def __mul__(self, v):
        if isinstance(v, vector):
            return self.dot(v)
    def __rmul__(self, s):
        return vector(s*self.entries)
    def __add__(self, v):
        return vector(self.entries + v.entries)
    def __sub__(self, v):
        return vector(self.entries - v.entries)
    def norm(self):
        return np.sqrt(self.entries.dot(self.entries.T))
    def dot(self, v):
        if isinstance(v, vector):
            return self.entries.dot(v.entries.T)
    
