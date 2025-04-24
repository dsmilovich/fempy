"""The file is organized into four tiny classes that each do one clear job:

1. Mesh             - holds the node coordinates.
2. FunctionSpace    - stores shape functions.
3. System           - builds element matrices/vectors and builds the linear system.
4. Solver           - solves the linear system with NumPy.
"""

import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import override


# -----------------------------------------------------------------------------
# 1) Mesh - describes the geometry (nodes)
# -----------------------------------------------------------------------------
class Mesh(ABC):
    def __init__(self):
        pass


class Mesh1D(Mesh):
    """Create a 1-D mesh from a list/array of x-coordinates."""

    def __init__(self, coordinates):
        # Convert any sequence to a NumPy array
        self.nodes = np.asarray(coordinates, dtype=float)

        # Check input validity (must be 1-D and sorted)
        if self.nodes.ndim != 1:
            raise ValueError("Coordinates must be a 1-D sequence.")
        if not np.all(np.diff(self.nodes) > 0):
            raise ValueError("Coordinates must be in ascending order.")


# -----------------------------------------------------------------------------
# 2) FunctionSpace - basis / shape functions
# -----------------------------------------------------------------------------
class FunctionSpace(ABC):
    """Declared abstract, does nothing, forces derived classes to implement these."""

    def __init__(self):
        pass

    @abstractmethod
    def ref_N(self, k, eta):
        pass

    @abstractmethod
    def ref_dN_deta(self, k, eta):
        pass

    @abstractmethod
    def N(self, e, k, x):
        pass

    @abstractmethod
    def dN_dx(self, e, k, x):
        pass


class FunctionSpaceSeg1(FunctionSpace):
    """Linear (P1) 1D-finite-element basis on a given mesh."""

    # Defines a quadrature rule for P1 1D elements ----------------------------
    class QuadratureRule1PointSeg1:
        """1-point Gauss rule on [-1, 1]."""

        def __init__(self):
            self.position = np.array([0.0])
            self.weight = np.array([2.0])
            self.n_quad = 1

    def __init__(self, mesh):
        # Mesh and element data
        self.mesh = mesh
        self.nodes = mesh.nodes
        self.n_nodes = len(self.nodes)
        self.n_elements = self.n_nodes - 1
        self.n_element_nodes = 2

        # Consecutive nodes form an element: (0,1), (1,2), ...
        self.elements = np.array([(i, i + 1) for i in range(self.n_elements)])
        self.element_sizes = np.array([self.nodes[j] - self.nodes[i] for (i, j) in self.elements])
        # x(eta) = x_a + 0.5 * h * (1 + eta), dx/deta = 0.5 * h
        self.J = 0.5 * self.element_sizes
        self.invJ = 1.0 / self.J

        # Attach quadrature rule
        self.quad_rule = self.QuadratureRule1PointSeg1()

    # Helper functions for assembling -----------------------------------------
    """Maps (element index, local node index) to global node index.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).

    Returns:
        int: Global node index corresponding to local index k in element e.
    """

    def local_to_global(self, e, k):
        return self.elements[e][k]

    """Maps (element index, global node index) to local node index.

    Args:
        e (int): Global element index.
        n (int): Global node index.

    Returns:
        int: Local node index (0 or 1 for a linear element) corresponding to global index n in element e.
    """

    def global_to_local(self, e, n):
        return n - e

    # Reference-element shape functions ---------------------------------------
    """Reference shape functions on [-1, 1]:
        N_0(eta) = (1 - eta) / 2,
        N_1(eta) = (1 + eta) / 2.

    Args:
        k (int): Local node index (0 or 1 for a linear element).
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: refN_k(eta).
    """

    @override
    def ref_N(self, k, eta):
        return ((1 - eta) * 0.5, (1 + eta) * 0.5)[k]

    """Derivatives of reference shape functions w.r.t. eta on [-1, 1]:
        dN_0/deta(eta) = -0.5,
        dN_1/deta(eta) = 0.5.

    Args:
        k (int): Local node index (0 or 1 for a linear element).
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: drefN_k/deta(eta).
    """

    @override
    def ref_dN_deta(self, k, eta):
        # Derivatives for the reference element are constant +/-0.5
        return (-0.5, 0.5)[k]

    # Element shape functions -------------------------------------------------
    """Return the value of the k-th (local index) 1-D linear shape function at
        global coordinate x for element of global index e.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).
        x (float): Global coordinate.

    Returns:
        float: N_{e,k}(x).
    """

    @override
    def N(self, e, k, x):
        pass

    """Return the value of the k-th 1-D linear shape function derivative w.r.t. x at
        global coordinate x for element of global index e.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).
        x (float): Global coordinate.

    Returns:
        float: dN_{e,k}/dx(x).
    """

    @override
    def dN_dx(self, e, k, x):
        pass


# TODO: implement this
class FunctionSpaceSeg2(FunctionSpace):
    """Quadratic (P2) 1D-finite-element basis on a given mesh."""

    # Defines a quadrature rule for P2 1D elements ----------------------------
    class QuadratureRule3PointSeg2:
        """3-point Gauss rule on [-1, 1]."""

        def __init__(self):
            pass

    def __init__(self, mesh):
        pass

    @override
    def local_to_global(self, e, k):
        pass

    @override
    def global_to_local(self, e, n):
        pass

    @override
    def ref_N(self, k, eta):
        pass

    @override
    def ref_dN_deta(self, k, eta):
        pass

    @override
    def N(self, e, k, x):
        pass

    @override
    def dN_dx(self, e, k, x):
        pass


# -----------------------------------------------------------------------------
# 3) System - problem definition
# -----------------------------------------------------------------------------
class System(ABC):
    """Build the global matrix A and RHS vector b from V."""

    def __init__(self, V):
        pass

    @abstractmethod
    def assemble_capacity(self):
        pass

    @abstractmethod
    def assemble_stiffness(self):
        pass

    @abstractmethod
    def assemble_residual(self):
        pass

    @abstractmethod
    def assemble_element_capacity(self, e):
        pass

    @abstractmethod
    def assemble_element_stiffness(self, e):
        pass

    @abstractmethod
    def assemble_element_residual(self, e):
        pass

    @abstractmethod
    def apply_BCs(self):
        pass

    @abstractmethod
    def apply_initial_conditions(self):
        pass


class SteadyStateHeatTransferSystem(System):
    """Build the stiffness matrix K and RHS vector q from V."""

    def __init__(self, V, k, Q):
        self.V = V

        # Conductivity and heat source/sink
        self.k = k
        self.Q = Q

        # Vector of unknowns
        self.u = np.zeros(self.V.n_nodes)

        # Global stiffness and RHS vector
        # TODO: implement a skyline sparse matrix storage scheme
        self.K = np.zeros((self.V.n_nodes, self.V.n_nodes))
        self.q = np.zeros(self.V.n_nodes)

        # Boundary condition types and values (0 = Neumann, 1 = Dirichlet)
        self.BC_types = np.zeros(self.V.n_nodes)
        self.BC_values = np.zeros(self.V.n_nodes)

    @override
    def assemble_capacity(self):
        pass

    @override
    def assemble_stiffness(self):
        for e in range(0, self.V.n_elements):
            self.assemble_element_stiffness(e)

        return

    @override
    def assemble_residual(self):
        for e in range(0, self.V.n_elements):
            self.assemble_element_residual(e)

        return

    @override
    def assemble_element_capacity(self, e):
        pass

    @override
    def assemble_element_stiffness(self, e):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local stiffness matrix
        local_K = np.zeros((n_element_nodes, n_element_nodes))

        for a in range(0, n_element_nodes):
            for b in range(0, n_element_nodes):
                for g in range(0, n_quad):
                    pos = self.V.quad_rule.position[g]
                    W = self.V.quad_rule.weight[g]
                    J = self.V.J[e]
                    dNa = self.V.ref_dN_deta(a, pos)
                    dNb = self.V.ref_dN_deta(b, pos)

                    local_K[a, b] += self.k * (dNa / J) * (dNb / J) * W * J

                i, j = self.V.local_to_global(e, a), self.V.local_to_global(e, b)
                self.K[i, j] += local_K[a, b]

        return

    @override
    def assemble_element_residual(self, e):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local RHS vector
        local_q = np.zeros(n_element_nodes)

        for a in range(0, n_element_nodes):
            for g in range(0, n_quad):
                pos = self.V.quad_rule.position[g]
                W = self.V.quad_rule.weight[g]
                J = self.V.J[e]
                Na = self.V.ref_N(a, pos)
                local_q[a] += Na * self.Q[self.V.local_to_global(e, a)] * W * J

            i = self.V.local_to_global(e, a)
            self.q[i] += local_q[a]

        return

    @override
    def apply_BCs(self):
        # Dirichlet BCs
        dirichlet_nodes = np.where(self.BC_types == 1)[0]

        for i in dirichlet_nodes:
            u_bar = self.BC_values[i]

            # Substract the i-th column times u_bar
            self.q -= self.K[:, i] * u_bar

            # Zero the i-th row/column
            self.K[i, :] = 0.0
            self.K[:, i] = 0.0

            # And set the corresponding diagonal value to 1, RHS to u_bar
            self.K[i, i] = 1.0
            self.q[i] = u_bar

        # Neumann BCs
        neumann_nodes = np.where(self.BC_types == 0)[0]

        for i in neumann_nodes:
            self.q[i] += self.BC_values[i]

    @override
    def apply_initial_conditions(self):
        pass


# TODO: implement this
class TransientHeatTransferSystem(SteadyStateHeatTransferSystem):
    """Build the capacity matrix, stiffness matrix K and RHS vector q from V."""

    def __init__(self, V, c, k, Q):
        pass

    @override
    def assemble_capacity(self):
        pass

    @override
    def assemble_stiffness(self):
        pass

    @override
    def assemble_residual(self):
        pass

    @override
    def assemble_element_capacity(self, e):
        pass

    @override
    def assemble_element_stiffness(self, e):
        pass

    @override
    def assemble_element_residual(self, e):
        pass

    @override
    def apply_BCs(self):
        pass

    @override
    def apply_initial_conditions(self):
        pass


# -----------------------------------------------------------------------------
# 4) LinearSolver - wraps NumPy's linear solver
# -----------------------------------------------------------------------------
class Solver(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def step(self, dt):
        pass


class SteadyStateSolver(Solver):
    def __init__(self, system):
        self.system = system

    @override
    def solve(self):
        # Assemble the stiffness matrix
        self.system.assemble_stiffness()
        # Assemble the RHS vector
        self.system.assemble_residual()
        # Apply the BCs
        self.system.apply_BCs()

        # And solve the system of linear equations
        # TODO: implement a CG solver that utilizes the skyline storage scheme
        self.system.u = np.linalg.solve(self.system.K, self.system.q)

        return

    @override
    def step(self, dt):
        pass


# TODO: implement this
class ForwardEulerSolver(Solver):
    def __init__(self):
        pass

    @override
    def solve(self):
        pass

    @override
    def step(self, dt):
        pass


# TODO: implement this
class BackwardEulerSolver(Solver):
    def __init__(self):
        pass

    @override
    def solve(self):
        pass

    @override
    def step(self, dt):
        pass


# -----------------------------------------------------------------------------
# Example run, change the parameters, function space, system, BCs and solver
# accordingly to generate different cases
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Definition of the geometry
    domain_length = 1.0
    n_nodes = 10

    # STEP 1: create geometry + basis
    x_coords = np.linspace(0.0, domain_length, n_nodes)  # np.linspace(start, stop, N)
    mesh = Mesh1D(x_coords)
    # modifiable
    V = FunctionSpaceSeg1(mesh)

    # STEP 2: define the system
    # Uniform conductivity, modifiable
    # TODO: make non-constant
    k = 1.0e-14
    # Heat source/sink
    # TODO: implement this
    Q = np.zeros(n_nodes)

    # Construct the system, modifiable (transient, etc.)
    system = SteadyStateHeatTransferSystem(V, k, Q)

    # Boundary conditions, modifiable
    # Dirichlet BCs at the endpoints
    system.BC_types[0] = 1
    system.BC_types[n_nodes - 1] = 1
    # Of values 1 and 2
    system.BC_values[0] = 1
    system.BC_values[n_nodes - 1] = 2

    # STEP 3: assemble and solve
    # Solver type is also modifiable
    solver = SteadyStateSolver(system)
    solver.solve()

    # print("Stiffness matrix K:")
    # print(system.K)
    # print("RHS vector q:")
    # print(system.q)
    # print("Solution vector u:")
    # print(system.u)

    # Plot the solution
    # TODO: comparison with the analytical solution
    plt.rcParams["font.size"] = 16
    plt.figure()
    plt.plot(V.nodes, system.u, marker="o", linewidth=2)
    plt.xlabel("$x$ $[m]$")
    plt.ylabel(r"Temperature $u$ $[°C]$")
    plt.title("1D steady-state temperature profile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
