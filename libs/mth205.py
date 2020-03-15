import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

def gs(basis):
    onbasis = []
    for b in basis:
        if len(onbasis) == 0: onbasis.append(unit(b))
        else: onbasis.append(unit(b-projection(b, onbasis)))
    return onbasis

def projection(b, basis):
    return np.sum([b*v/(v*v)*v for v in basis])

def unit(v):
    return 1/v.norm()*v

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
                self.entries = np.array(columns)
            else:                         ## numpy array
                self.entries = args[0]
            shape = np.shape(self.entries)
            self.rows = shape[0]
            self.cols = shape[1]
        self.entries = self.entries.astype('float64')
        if transpose:
            self.T = matrix(self.entries.T, False)
            self.T.T = self

    def column(self, c):
        return vector(self.entries[:, c])
    def row(self, r):
        return vector(self.entries[r, :])
    def matrix_from_columns(self, c):
        columns = [self.entries[:, col] for col in c]
        return matrix(np.array(columns).T)
    def columns(self):
        return [self.column(j) for j in range(self.cols)]
    def matrix_from_rows(self, r):
        rows = [self.entries[row, :] for row in r]
        return matrix(np.array(rows))
    def matrix_from_rows_and_columns(self, r, c):
        return self.matrix_from_rows(r).matrix_from_columns(c) 
    def eigenvalues(self):
        return LA.eig(self.entries)[0]
    def right_eigenmatrix(self):
        values, vectors = LA.eig(self.entries)
        order = np.argsort(values)[::-1]
        d = np.diag(values[order])
        '''
        d = np.zeros((len(values), len(values)))
        for i, v in enumerate(values):
            d[i][i] = v
        '''
        return (matrix(d), matrix(vectors[order]))
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

            index = np.argmax(np.abs(column)) + row
            if A[index, p] == 0:
                continue
            pivots.append(p)
            if index != row:
                rowi = np.copy(A[index])
                A[index] = A[row]
                A[row] = rowi
            m = A[row,p]
            A[row] = 1/A[row, p] * A[row]
            A[row,np.abs(A[row, :]) < 1e-10] = 0
            for r in range(row+1, self.rows):
                A[r] -= A[r,p]*A[row]
                A[r,np.abs(A[r, :]) < 1e-10] = 0
            if pivots[-1] == self.rows - 1:
                break
        pivots.reverse()
        rank = len(pivots)
        for i, p in enumerate(pivots):
            row = rank - i - 1
            for j in range(row):
                A[j] -= A[j,p] * A[row]
                A[j, np.abs(A[j, :]) < 1e-10] = 0
        return matrix(A)

    def right_kernel(self):
        tolerance = 1e-14
        rref = self.rref().entries
        basic = []
        for row in rref:
            if (np.abs(row) > tolerance).any(): 
                basic.append(np.argwhere(row != 0)[0][0])
            else:
                break
        free = list(set(range(self.cols)).difference(basic))
        basis = []
        for f in free:
            b = np.zeros(self.cols)
            b[f] = 1
            col = rref[:, f]
            for i, c in np.ndenumerate(basic):
                b[c] = -col[i]
            basis.append(vector(b))
        return basis

    def inverse(self):
        return matrix(LA.inv(self.entries))

    def augment(self, b):
        if isinstance(b, vector):
            b = b.entries.reshape(len(b.entries), 1)
            return matrix(np.hstack([self.entries, b]))
        return matrix(np.hstack([self.entries, b.entries]))

    def transpose(self):
        return matrix(self.entries.T)

    def display(self, scale=1):
        '''
        from matplotlib.colors import LinearSegmentedColormap
        map_colors=[(1,0,0), (0,0,0), (1,1,1)]
        cm = LinearSegmentedColormap.from_list("my_list", map_colors, N=100)
        '''
        
        entries = np.copy(self.entries)
        shape = entries.shape
        max = np.max(np.abs(entries))
        entries = 1/max*entries
        fig, ax = plt.subplots(figsize=(scale*shape[1], scale*shape[0]))
        #plt.imshow(entries, cmap=cm)
        plt.imshow(entries)
        ax.set_aspect(1)
        plt.xticks(range(shape[1]))
        plt.yticks(range(shape[0]))
        plt.colorbar(orientation="vertical")
        plt.clim(-1,1)
        plt.show()

    def lu(self):
        return map(matrix, LA.lu(self.entries))

    def QR(self):
#        Q = matrix(gs([self.column(c) for c in range(self.cols)])).T
#        R = Q.T*self

        q, r = LA.qr(self.entries)
        Q = matrix(q)
        R = matrix(r)
        return Q, R

    def copy(self):
        return matrix(np.copy(self.entries))

    def printf(self, decimals=3):
        np.set_printoptions(precision = decimals, suppress=True)
        print(self.entries)

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
    def __repr__(self):
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
    def __repr__(self):
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
    def copy(self):
        return vector(np.copy(self.entries))
    def norm(self):
        return np.sqrt(self.entries.dot(self.entries.T))
    def dot(self, v):
        if isinstance(v, vector):
            return self.entries.dot(v.entries.T)
    def display(self):
        matrix([self]).display()
    def demean(self):
        return vector(self.entries - self.entries.mean())
    def dim(self):
        return len(self.entries)
    def __getitem__(self, n):
        return self.entries[n]
    def printf(self, decimals=3):
        np.set_printoptions(precision = decimals, suppress=True)
        print(self.entries.T)

def list_plot(data, color="blue", aspect_ratio=None, size=25,
              ylim = None,
              dims=(8,6), title=None):
    if isinstance(data, matrix):
        entries = data.entries
    else:
        entries = np.array([d.entries for d in data]).T
    if dims != None:
        fig, ax = plt.subplots(figsize=dims)
    else:
        fig, ax = plt.subplots()
    if aspect_ratio != None:
        ax.set_aspect(aspect_ratio)
    if title != None:
        ax.set_title(title)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.scatter(x=entries[0], y=entries[1], c=color, s=size)

def mean(data):
    if isinstance(data, list) and isinstance(data[0], vector):
        entries = np.array([v.entries for v in data]).T
        return vector(np.mean(entries, axis=1))
    if isinstance(data, vector):
        return np.mean(data.entries)
    return np.mean(data)

def onesvec(n):
    return vector(np.ones(n))

def zerovec(n):
    return vector(np.zeros(n))

def plot_model(xhat, data, color='blue',
               aspect_ratio = None,
               title = None,
               ylim = None,
               size=25, dims=(8,6)):
    entries = np.array([d.entries for d in data]).T
    max = np.max(entries[0])
    min = np.min(entries[0])
    plotx = np.linspace(num=100, start=min, stop=max)
    k = len(xhat.entries)
    ploty = np.array([vector([x**j for j in range(k)])*xhat for x in plotx])
    if dims != None:
        fig,ax = plt.subplots(figsize=dims)
    else:
        fig,ax = plt.subplots()
    if aspect_ratio != None:
        ax.set_aspect(aspect_ratio)
    if title != None:
        ax.set_title(title)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.scatter(x = entries[0], y = entries[1], c = color, s =size)
    ax.plot(plotx, ploty, color='red')

def vandermonde(data, k):
    return matrix([ vector([x**j for j in range(k+1)]) for x in data.entries])




