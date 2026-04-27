from multiprocessing import Value
from operator import ne

import pymanopt as mo
import pymanopt.manifolds as man
from pymanopt.manifolds.manifold import Manifold
import numpy as np
import numpy.linalg as lg
from dataclasses import dataclass
import scipy.linalg
import scipy.stats
import scipy
import networkx as nx


@dataclass
class ProblemSynchRotations:
    """ """

    d: int
    n: int = 2
    p: float = 1.0  # erdos reyni probability
    eps: float = 0.0  # noise
    seed: int = 0
    warmstart: float = 1.0  # role of a step

    def __post_init__(self):
        np.random.seed(self.seed)
        self.mdim = (self.d * (self.d - 1) // 2) * self.n
        self.adim = self.n * (self.d**2)
        self.codim = self.adim - self.mdim
        manif = man.Product(
            [mo.manifolds.SpecialOrthogonalGroup(self.d) for _ in range(self.n)]
        )

        self.manifold = manif
        self.xstar = self.manifold.random_point()
        G = nx.erdos_renyi_graph(self.n, self.p)
        self.G = list(G.edges())
        self.H = [self.xstar[i].dot(self.xstar[j].T) for (i, j) in self.G]
        self.H = [
            self.H[i] + self.eps * np.random.randn(*self.H[0].shape)
            for i in range(len(self.H))
        ]  # adding noise

        if self.warmstart != 0:
            random_tg = self.manifold.random_tangent_vector(self.xstar)

            self.xstart = self.manifold.exp(self.xstar, self.warmstart * random_tg)
        else:
            self.xstart = self.manifold.random_point()
        self.fstart = self.costf(self.xstart)

    def anorm(self, v):
        return np.linalg.norm(
            np.array([np.linalg.norm(v[i], ord="fro") for i in range(self.n)]), ord=2
        )

    def stack(self, x):
        """stacks point on the manifold to tensor types"""
        return np.stack(x)  # one single tensor

    def unstack(self, x):
        return tuple(
            x[i] for i in range(x.shape[0])
        )  # convert points back to pymanopt product type

    def costf(self, x):
        sum = 0
        for ind, (i, j) in enumerate(self.G):
            sum = sum + np.linalg.matrix_norm(x[i] - self.H[ind].dot(x[j])) ** 2
        return sum

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
        xs = self.stack(x)
        if projection == 1:
            abase = np.eye(self.adim)
            # application de la rotation avant de reshaper : vérifier que ça fait du sens !
            rotmat = scipy.stats.ortho_group.rvs(self.adim)
            if rotation == 1:
                abase = rotmat.dot(abase)
            abase = abase.reshape(self.adim, self.n, self.d, self.d)
            # les vecteurs de taille (self.n,self.d,self.d) sont du type du l'espace ambiant, à unstack près.

            # Defining the ambient pss
            if psstype == 1:
                apss = np.concat((abase, -abase), axis=0)
            elif psstype == 2:
                negatedsum = -np.sum(abase, axis=0)
                negatedsumnorm = np.sqrt(np.sum(negatedsum**2))
                negatedsum = negatedsum / negatedsumnorm
                negatedsum = negatedsum.reshape(1, self.n, self.d, self.d)
                apss = np.concatenate((abase, negatedsum), axis=0)

            elif psstype == 3:
                GramMatrix = (-1 / self.adim) * np.ones((self.adim, self.adim)) + (
                    1 + 1 / self.adim
                ) * np.eye(self.adim)

                L = scipy.linalg.cholesky(GramMatrix)
                apss = np.einsum(
                    "ijkl,im->mjkl", abase, L.T
                )  # lines of L have uniform angles, so transpose it to use these lines as coefficients to combine the base with.
                negatedsum = -np.sum(apss, axis=0)
                negatedsum = negatedsum.reshape(1, self.n, self.d, self.d)
                apss = np.concatenate((apss, negatedsum), axis=0)
            else:
                raise ValueError("psstype should be either 1, 2 or 3")

            # projecting the pss (see Boumal, p162)

            xTu = np.einsum("nik,bnkj->bnij", np.transpose(xs, axes=(0, 2, 1)), apss)
            uTx = np.einsum("bnik,nkj->bnij", np.transpose(apss, axes=(0, 1, 3, 2)), xs)
            antisymetrized = (xTu - uTx) / 2
            pss = np.einsum("nik,bnkj->bnij", xs, antisymetrized)

            # renormalizing the projected vectors (embedded subm)
            pssnorms = np.sqrt(np.sum(pss**2, axis=(1, 2, 3)))
            pssnormskeep = np.where(pssnorms >= tolerance)
            pssnormsdiscard = np.where(pssnorms < tolerance)
            nblostvec = len(pssnormsdiscard)

            pss = pss[pssnormskeep]
            pssnorms = pssnorms[pssnormskeep]
            pss = pss / pssnorms[:, np.newaxis, np.newaxis, np.newaxis]
            # psspymanopt =

        elif projection == 0:
            # build base of tangent space
            def antisymmetric_basis(n):
                i, j = np.triu_indices(n, k=1)
                k = len(i)

                B = np.zeros((k, n, n))
                B[np.arange(k), i, j] = 1
                B[np.arange(k), j, i] = -1

                return B / np.sqrt(2)

            def product_antisymmetric_basis(d, n):
                B = antisymmetric_basis(d)  # (d, n, n)
                w = B.shape[0]

                basis = np.zeros((n, w, n, d, d))
                basis[np.arange(n), :, np.arange(n), :, :] = B
                return basis.reshape(n * w, n, d, d)

            base = product_antisymmetric_basis(self.d, self.n)

            # application de la rotation (comb lin de la base)
            rotmat = scipy.stats.ortho_group.rvs(self.mdim)
            if rotation == 1:
                base = np.einsum("kabc,kd->dabc", base, rotmat)

            # multiplying by x at left to finally have an orthogonal basis of the tangent space
            base = np.einsum("nik,bnkj->bnij", xs, base)

            if psstype == 1:
                pss = np.concat((base, -base), axis=0)
            elif psstype == 2:
                negatedsum = -np.sum(base, axis=0)
                negatedsumnorm = np.sqrt(np.sum(negatedsum**2))
                negatedsum = negatedsum / negatedsumnorm
                negatedsum = negatedsum.reshape(1, self.n, self.d, self.d)
                pss = np.concatenate((base, negatedsum), axis=0)
            elif psstype == 3:
                GramMatrix = (-1 / self.mdim) * np.ones((self.mdim, self.mdim)) + (
                    1 + 1 / self.mdim
                ) * np.eye(self.mdim)

                L = scipy.linalg.cholesky(GramMatrix)
                pss = np.einsum(
                    "ijkl,im->mjkl", base, L.T
                )  # lines of L have uniform angles, so transpose it to use these lines as coefficients to combine the base with.
                negatedsum = -np.sum(pss, axis=0)
                negatedsum = negatedsum.reshape(1, self.n, self.d, self.d)
                pss = np.concatenate((pss, negatedsum), axis=0)
            else:
                raise ValueError("psstype should be either 1, 2 or 3")
        else:
            raise ValueError("projection should be either 0 or 1")
        pssbis = []
        for i in range(pss.shape[0]):
            pssbis.append(self.unstack(pss[i]))
        pss = pssbis
        return pss
