import pymanopt as mo
import numpy as np
import numpy.linalg as lg
import pymanopt.manifolds as man
from dataclasses import dataclass
from pymanopt.manifolds.manifold import Manifold
import scipy.linalg
import scipy.stats
import scipy
from influence_of_dimension.tools.utils_cm_hypersphere import *
from tools.problems.newmanifolds import NullManifold


@dataclass
class ProblemSynchRotations:
    """Build a minimization problem on a sphere in R^n, with all required methods to run the directsearch.
    Args:
        adim (int) : ambient dimension
        mdim (int) : manifold dimension. mdim>=2.
    Note:
        The manifold is a sphere of dimension mdim embedded in R^{mdim+1}, concatenated with a null manifold of dimension adim - (mdim+1).
    """

    adim: int
    mdim: int = 

    def __post_init__(self):
        self.codim = self.adim - self.mdim
        self.manifold = man.Product(
            [
                mo.manifolds.Sphere(self.mdim + 1),
                NullManifold(man.Euclidean(self.codim - 1)),
            ]
        )
        self.anorm = lambda x: np.sqrt(
            scipy.linalg.norm(x[0], ord=2) ** 2 + scipy.linalg.norm(x[1], ord=2) ** 2
        )  # ambient norm = euclidean norm since embedded submanifold
        A = np.random.randn(self.mdim + 1, self.mdim + 1)
        self.A = (A + A.T) / 2
        self.xstart = self.manifold.random_point()
        self.fstart = self.costf(self.xstart)

    def costf(self, x):
        value = np.dot(self.A, x[0]).dot(x[0])
        return value

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
            pss (list) : list of vectors of the tangent space (compatible with the manifold type), forming a positive spanning set.
        """


        if projection == 1:
            abase_U = np.eye(self.mdim + 1)
            abase_V = np.eye(self.codim - 1)
            abase = []
            for vec in abase_U:
                abase.append(np.hstack((vec, np.zeros(self.codim - 1))))
            for vec in abase_V:
                abase.append(np.hstack((np.zeros(self.mdim + 1), vec)))
            abase = np.array(abase).T  # each column is a vector

            # applying rotation to basis
            if rotation == 1:
                rotmat = scipy.stats.ortho_group.rvs(self.adim)
                base = abase.dot(rotmat)
            else:
                base = abase

            # Defining the ambient pss
            if psstype == 1:
            elif psstype == 2:
            elif psstype == 3:

            # projecting the pss
            pss = []
            for vec in apss.T:
                pymanoptvec = [vec[: self.mdim + 1], vec[self.mdim + 1 :]]
                pvec = self.manifold.projection(x, pymanoptvec)
                # removing zero vectors
                pvecnorm = self.manifold.norm(x, pvec)
                if pvecnorm > tolerance:  # removing zero vectors
                    if renormalize == True:  # renormalizing nonzero vectors
                        pvec = pvec / pvecnorm
                    pss.append(pvec)
            nblostvec = len(apss.T) - len(pss)
            pymanoptpss = pss

            # remark : if no rotation was applied, then many vectors are aligned with the axes. After projection, they are removed. The projected method will thus have roughly as many vectors as the intrinsic one.

        elif projection == 0:


        if returnlostvecnumber == 1:
            return pymanoptpss, nblostvec
        else:
            return pymanoptpss
