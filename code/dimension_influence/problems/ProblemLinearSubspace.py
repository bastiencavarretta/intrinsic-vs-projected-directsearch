import numpy as np
import scipy
from dataclasses import dataclass
from problems.newmanifolds import LinearSubspace


@dataclass
class ProblemLinearSubspace:
    """
    Build a minimization problem on a linear subspace of R^n, with all required methods to run the directsearch.

    Args:
        B (np.ndarray) : Stiefel matrix of shape (n,m). m must be larger than 2 and n.
        obj (int) : 1 or 2 or 3
    Note:
        The linear subspace is defined as the image of the Stiefel matrix B.
        If obj == 1, the problem corresponds to the barycenter of points in the linear subspace.
        If obj == 2, the problem corresponds to the barycenter of points in the ambient space R^n (constrained to the linear subspace).
        If obj == 3, the objective is a quadratic function.
    """

    B: np.ndarray  # Stiefel matrix of shape (n,m). m must
    obj: int = 1

    def __post_init__(self):  # adim = ambient dim
        self.manifold = LinearSubspace(self.B)
        self.adim = self.B.shape[0]
        self.mdim = self.manifold.dimension
        self.codim = self.adim - self.mdim
        self.xstart = self.manifold.random_point()

        if self.obj == 1:  # Barycenter problem of points in the ambient subspace R^n
            self.xref = np.random.randn(self.adim)
            self.xrefdirections = [np.random.rand(self.adim) for i in range(10)]
            self.xpoints = [self.xref + v for v in self.xrefdirections]
            self.xstar = self.manifold.projection( 
                1967, np.mean(np.array(self.xpoints), axis=0)
            ) # Linear subspace: the projection formula does not depend on the current point. First argument useless
        elif self.obj == 2:  # Barycenter problem of points in the linear subspace
            self.xref = self.manifold.random_point()
            self.xrefdirections = [
                self.manifold.random_tangent_vector(self.xref) for i in range(10)
            ]
            self.xpoints = [
                self.manifold.exp(self.xref, v) for v in self.xrefdirections
            ]
            self.xstar = np.mean(np.array(self.xpoints), axis=0)

        elif self.obj == 3:  # Quadratic objective function
            rdmmatrix = np.random.randn(self.adim, self.adim)
            self.A = rdmmatrix.dot(rdmmatrix.T) + 0.1 * np.eye(self.adim, dtype=float)
            self.b = np.random.randn(self.adim)
        else:
            raise ValueError("objective not defined")

        self.fstart = self.costf(self.xstart)

    def anorm(self, x):
        return scipy.linalg.norm(x, ord=2)

    def costf(self, x):
        if self.obj == 1:
            cost = 0
            for point in self.xpoints:
                cost = cost + self.manifold.dist(point, x) ** 2
            cost = cost / len(self.xpoints)
            return cost

        elif self.obj == 2:
            cost = 0
            for point in self.xpoints:
                cost = cost + np.linalg.norm(x - point, ord=2) ** 2
            cost = cost / len(self.xpoints)
            return cost
        elif self.obj == 3:
            return ((self.A).dot(x).dot(x)) / 2 - (self.b).dot(x)
        else:
            raise ValueError("non defined objective")

    def build_pss(
        self,
        x,
        projection=0,
        psstype=1,
        rotation=0,
        renormalize=True,
        tolerance=1e-16,
        returnlostvecnumber=0,
    ):
        """Building the positive spanning set of the tangent space at point x
        Args:
            x : point on the manifold
            projection (int) : 0 or 1. Whether the pss is intrinsic (0) or projected (1)
            psstype (int) : 1,2 or 3. Type of positive spanning set to build (same order as table 1 in the paper)
            rotation (int) : 0 or 1. Whether to apply a random rotation to the basis before building the pss
            renormalize (bool) : whether to renormalize the vectors of the pss after projection
            tolerance (float) : tolerance to remove small "null" vectors after projection
            returnlostvecnumber (int) : 0 or 1. Whether to return the number of lost vectors after projection or not (for test purposes only)
        Returns :
            pss (np.ndarray) : array of shape (nb_vectors, dimension of the ambient space). Each row is a vector of the positive spanning set.
        """
        if projection == 1:  # projected pss
            base = np.eye(self.adim)
            # applying rotation to basis
            if rotation == 1:
                rotmat = scipy.stats.ortho_group.rvs(self.adim)
                base = base.dot(rotmat)
            # defining the ambient pss
            if psstype == 1:
                pss = np.concatenate((base, -base), axis=1)
            elif psstype == 2:
                negatedsum = -np.sum(base, axis=1).reshape(-1, 1)
                negatedsum = negatedsum / np.linalg.norm(negatedsum, ord=2)
                pss = np.concatenate((base, negatedsum), axis=1)
            elif psstype == 3:
                GramMatrix = (-1 / self.adim) * np.ones((self.adim, self.adim)) + (
                    1 + 1 / self.adim
                ) * np.eye(self.adim)
                L = scipy.linalg.cholesky(GramMatrix)
                pss = base.dot(L)
                negatedsum = -np.sum(pss, axis=1).reshape(-1, 1)
                pss = np.concatenate((pss, negatedsum), axis=1)

            # projecting the pss
            pss = np.array(
                [self.manifold.projection(x, pss[:, i]) for i in range(pss.shape[1])]
            ).T

            # removing zero vectors
            pssstar = []
            for vec in pss.T:
                if self.manifold.norm(x, vec) > tolerance:
                    pssstar.append(vec)
            pssstar = np.array(pssstar).T
            nblostvec = pss.shape[1] - pssstar.shape[1]
            pss = pssstar

            # renormalizing
            if renormalize == True:
                pss = pss / np.linalg.norm(pss, ord=2, axis=0, keepdims=True)

        else:  # intrinsic pss
            base = self.B.copy()
            # applying rotation to basis
            if rotation == 1:
                rotmat = scipy.stats.ortho_group.rvs(self.mdim)
                base = base.dot(rotmat)
            # defining the pss
            if psstype == 1:
                pss = np.concatenate((base, -base), axis=1)
            elif psstype == 2:
                negatedsum = -np.sum(base, axis=1).reshape(-1, 1)
                negatedsum = negatedsum / self.manifold.norm(x, negatedsum)
                pss = np.concatenate((base, negatedsum), axis=1)
            elif psstype == 3:
                GramMatrix = (-1 / self.mdim) * np.ones((self.mdim, self.mdim)) + (
                    1 + 1 / self.mdim
                ) * np.eye(self.mdim)
                L = scipy.linalg.cholesky(GramMatrix)
                pss = base.dot(L)
                negatedsum = -np.sum(pss, axis=1).reshape(-1, 1)
                pss = np.concatenate((pss, negatedsum), axis=1)
        if returnlostvecnumber == 1:
            return pss.T, nblostvec
        else:
            return pss.T
