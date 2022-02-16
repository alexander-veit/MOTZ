# MOTZ (Marching-Of-The-Zeros)

This repository implements the algorithms `MOTZ` and `MOTZ_flip` developed in 

M. Bernkopf, S. Sauter, C. Torres, A. Veit. Solvability of Discrete Helmholtz Equations. To appear in IMA Journal of Numerical Analysis.

The easiest way to get started is to open the Jupyter notebook and look at the provided examples. The notebook provides two example meshes - one where `MOTZ` returns `certified` and one where `critical` is returned. We are using the python libraries `dmsh` and `meshplex` for mesh generation. The second example shows how the correct mesh input format can be achieved by providing nodes and cells (triangles) of a mesh and should therefore be straighforward to adapt to a mesh of interest.
