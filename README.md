This package is meant for students taking linear algebra classes at [Grand Valley State University](https://gvsu.edu/math) though others are certainly welcome to use it.  Students become comfortable using Sage to do some linear algebraic computations but eventually start working in Colab notebooks.  To help facilitate working in a Colab notebook, this package provides access to `numpy` commands using a Sage-like syntax.  There are a few other features for visualizing the results of some computations.

Within a Colab notebook or other pure Python environment, begin with

```
!pip install gv_linalg
from gv_linalg import *
```
Then Sage-like commands, such as,

```
A = matrix([[1,2],[2,1]])
A.eigenvalues()
```
work as expected.