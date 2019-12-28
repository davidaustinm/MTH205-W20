import numpy as np
import numpy.linalg as LA

class matrix:
    def __init__(self, *args):
        transpose = True
        if len(args) == 2:
            transpose = args[1]
            args = args[:1]
        if len(args) == 3:
            rows = args[0]
            cols = args[1]
            entries = args[2]
            self.entries = np.reshape(entries, (rows, cols))
            self.rows = rows
            self.cols = cols
        if len(args) == 1:
            if isinstance(args[0],list):  ## list of vectors
                columns = [v.entries for v in args[0]]
                self.entries = np.array(columns).T
            else:                         ## numpy array
                self.entries = args[0]
            shape = np.shape(self.entries)
            self.rows = shape[0]
            self.cols = shape[1]
        self.entries = self.entries.astype('float64')
        if transpose:
            self.T = matrix(self.entries.T, False)
            self.T.T = self

    def getcolumns(self, c):
        if isinstance(c, list):
            columns = [self.entries[:, col] for col in c]
            return matrix(np.array(columns).T)
        return vector(self.entries[:, c])
    def getrows(self, r):
        if isinstance(r, list):
            rows = [self.entries[row, :] for row in r]
            return matrix(np.array(rows))
        return vector(self.entries[r, :])
    def eigenvalues(self):
        return LA.eig(self.entries)[0]
    def right_eigenmatrix(self):
        values, vectors = LA.eig(self.entries)
        return (values, matrix(vectors))
    def det(self):
        return LA.det(self.entries)
    def determinant(self):
        return self.det()
    def rref(self):
        ## only for teaching purposes
        pivots = []
        A = np.copy(self.entries)
        for p in range(self.cols):  # look for pivot in column i
            row = len(pivots)
            column = A[row:, p]
            index = np.argmax(column) + row
            if A[index, p] == 0:
                continue
            pivots.append(p)
            if index != row:
                rowi = np.copy(A[index])
                A[index] = A[row]
                A[row] = rowi
            A[row] = 1/A[row, p] * A[row]
            A[row,np.abs(A[row, :]) < 1e-10] = 0
            for r in range(row+1, self.rows):
                A[r] -= A[r,p]*A[row]
            if pivots[-1] == self.rows - 1:
                break
        pivots.reverse()
        print(pivots)
        print (A)
        rank = len(pivots)
        for i, p in enumerate(pivots):
            row = rank - i - 1
            print(row, p)
            for j in range(row):
                A[j] -= A[j,p] * A[row]
                A[j, np.abs(A[j, :]) < 1e-10] = 0
        return matrix(A)

    def inverse(self):
        return matrix(LA.inv(self.entries))

    def augment(self, b):
        if isinstance(b, vector):
            b = b.entries.reshape(len(b.entries), 1)
            return matrix(np.hstack([self.entries, b]))
        return matrix(np.hstack([self.entries, b.entries]))

    def transpose(self):
        return matrix(self.entries.T)
        
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

    def __xor__(self, n):
        if isinstance(n, int):
            A = np.copy(self.entries)
            if n < 0:
                A = LA.inv(A)
                n = np.abs(n)
            A = matrix(A)
            if self.rows == self.cols:
                B = matrix(np.identity(self.rows))
            for i in range(n):
                B = A*B
            return B
            
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



