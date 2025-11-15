import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm #colormaps
import scipy.linalg as lg

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import binom
import itertools
import pandas as pd
import pymanopt

from tqdm.notebook import tqdm
from matplotlib import colormaps as cmap
plt.ion()



def normalize(vectors, tol = 10**-15):
    """
    Normalizing a set of vectors, and identifying the null ones (below tol).
    Args :
        vectors (np.ndarry): n py m matrix whose columns are the vectors to normalize.
        tol (float): tolerance below which a vector is considered null.
    Returns :
        normalized_vectors (np.ndarray) : n by p matrix (p<=m) whose columns are the normalized vectors (or 0 if the norm was < tol)
        indices_zero (np.ndarray) : array of indices related to vectors set to zero
        indices_keep (np.ndarray) : array of indices related to vectors kept.
    """
    dim = vectors.shape[0]
    norms = lg.norm(vectors,axis = 0)
    indices_zero = np.where(norms < tol)[0]
    indices_keep = np.where(norms >= tol)[0]
    norms[indices_zero] = 1 # to avoid division by zero
    normalized_vectors = vectors/norms
    normalized_vectors[:,indices_zero] = np.zeros((dim, len(indices_zero)))   # withdrawing small vectors
    return normalized_vectors,indices_zero, indices_keep

def compute_cm_HJB(M,tol = 1e-15):
    """
    Computing the cosine measure of a positive spanning set using the algorithm from :
    "W. Hare, G. Jarry-Bolduc, 2020, A deterministic algorithm to compute the cosine measure of a finite positive spanning set" 

    Args : 
        M (np.ndarray) : n by p matrix whose columns are the vectors of the PSS (dimension is thus n)
        tol (float) : tolerance for nonsingularity of maximal submatrices in M
        
    Returns :
        cm (float) :the cosine measure of this PSS
    """
    list_potential_cm = [] #list of \dotted p_B
    n = M.shape[0]
    for family in itertools.combinations(M.T, n):
        B = np.array(family).T 
        if np.abs(np.linalg.det(B)) >= tol : # B must be a base
            G_B = np.dot(B.T,B)   # the Gram matrix.
            gamma_B = 1/np.sqrt(np.ones(n)@lg.solve(G_B,np.ones(n))) # more efficient : Cholesky and solve triangular systems
            u_B = gamma_B*lg.solve(B.T,np.ones(n))
            p_B = u_B.T@M
            p_B_zero = np.max(p_B)
            list_potential_cm.append(p_B_zero)
    cm = np.min(list_potential_cm)
    return cm



def generate_PSS(n,psstype= 1):
    """Generate an ambiant PSS in R^n of given type.
    Args :
        n (int): ambient dimension
        psstype (int): 1 or 2 or 3. type of the pss we want to generate. (same order as table 1 in the paper)
    Returns :
        pss (ndarray) : n by k matrix whose columns are the vectors of the PSS
    """
    if psstype == 1: 
        pss = np.zeros((n,2*n))
        pss [:,:n] = np.eye(n)
        pss [:,n:] = -np.eye(n) 
        
    elif psstype == 2: 
        pss = np.eye(n,n+1)
        pss [:,-1] = -np.ones(n)/np.sqrt(n)
    
    elif psstype == 3:
        A = -np.ones((n,n))/n + (1+1/n) * np.eye(n)
        L = lg.cholesky(A)
        pss = np.zeros((n,n+1))
        pss[:,:-1] = L
        pss[:,-1] = -np.sum(L, axis =1)
    else :
        raise ValueError("wrong 'ambiant_pss_type' specified")  
    pss,_,_ = normalize(pss)
    return(pss)


# Objective function to solve the cosine measure problem of the projected pss 1 on tangent space of the sphere
# def w_square(x,psstype = 1, tol_HJB = 1e-14, tol = 1e-15):
#     n = len(x)
#     sphere = pymanopt.manifolds.Sphere(n)
#     pss = generate_PSS(n,psstype = psstype)

#     # construire un BON de T_xM
#     matrix = np.eye(n)
#     matrix[:,0] = x 
#     Q,_ = lg.qr(matrix)
#     base_tangentspace = Q[:,1:]
        

#     # construction et normalisation de pss proj 
#     pss_proj = np.array([sphere.to_tangent_space(x, vec) for vec in pss.T]).T
#     normalized_vectors,indices_zero, indices_keep = normalize(pss_proj, tol = tol)
#     pss_proj = normalized_vectors[:,indices_keep]

#     # décomposition des vecteurs de pss_proj dans cette base : matrice à m lignes
#     pss_proj_base_tangentspace = base_tangentspace.T.dot(pss_proj)


#     # Boucle de Hare/Jarry-Bolduc
#     value = 1/compute_cm_HJB(pss_proj_base_tangentspace, tol = tol_HJB)

#     return value


def CM_heatmap_spheredim2(psstype = 1,
               n_samples = 1000,
               min_theta=  -0.1 ,  max_theta = 2*np.pi + 0.1,
               min_phi = -0.1, max_phi  = np.pi + 0.1,
               tol = 10**(-10), return_intrinsic_cm = False,tol_HJB = 1e-15):
    """Compute the cosine measure of projected PSS on the sphere of dimension 2 on a grid of points in spherical coordinates.
    Args :
        psstype (int): 1 or 2 or 3. type of the pss (same order as table 1 in the paper).
        n_samples (int): approximate number of sample points on the sphere.
        min_theta (float): minimum longitude angle.
        max_theta (float): maximum longitude angle.
        min_phi (float): minimum latitude angle (0 = north pole).
        max_phi (float): maximum latitude angle (pi = south pole).
        tol (float): tolerance for the renormalization of vectors (see normalize function).
        return_intrinsic_cm (boolean): whether to return the intrinsic cosine measure heatmap as well.
        tol_HJB (float): tolerance for the compute_cm_HJB function.
    Returns :
        V_theta (ndarray): meshgrid of longitude angles.
        V_phi (ndarray): meshgrid of latitude angles.
        CM_proj (ndarray): cosine measure heatmap of the projected PSS.
        X (ndarray): meshgrid of points on the sphere.
        pss_ambiant (ndarray): ambiant PSS used for the projection.
        CM_intr (ndarray): cosine measure heatmap of the intrinsic PSS (if return_intrinsic_cm = True)."""


    m,n = 2,3
    # Meshing the sphere 
    N_phi = int(np.sqrt(n_samples)) 
    N_theta =  int(np.sqrt(n_samples))
    v_theta = np.linspace(min_theta, max_theta,N_theta)
    v_phi = np.linspace(min_phi,max_phi,N_phi)
    V_theta,V_phi = np.meshgrid(v_theta,v_phi)
    X = np.zeros((N_theta,N_phi,n)) # actual mesh on the sphere (will store the value of $x$ for every angles)
    
    # Generating ambient PSS
    pss_ambient = generate_PSS(n,psstype = psstype)

    CM_proj = np.zeros((N_theta,N_phi))
    CM_intr = np.zeros((N_theta,N_phi))
    for i in range(N_theta):
        for j in range(N_phi):
            theta = v_theta[i]
            phi = v_phi[j]
            x = np.array([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)]).T
            # Orthogonal basis of tangent space
            e_theta = np.array([[-np.sin(theta), np.cos(theta), 0.]]).reshape(-1,1)
            e_phi = np.array([[-np.cos(phi)*np.cos(theta),-np.cos(phi)*np.sin(theta),np.sin(phi)]]).reshape(-1,1)
            base_tangentspace = np.concatenate((e_theta,e_phi), axis = 1)
            X[i,j] = x # store the values of $x$, to draw the heatmap later on

            # projected pss
            proj = lambda v :  v - np.tensordot(np.matmul(x.T,v),x,axes = 0).T  
            pss_proj = proj(pss_ambient)
            pss_proj_normalized,indices,indices_keep = normalize(pss_proj, tol = tol)
            pss_proj = pss_proj_normalized[:,indices_keep] 
            
            # intrinsic pss 
            pss_intr = base_tangentspace.dot(generate_PSS(2,psstype=psstype)) 
            
            # decompose the pss in the orthogonal basis 
            pss_proj_base = base_tangentspace.T.dot(pss_proj)
            pss_intr_base = base_tangentspace.T.dot(pss_intr)

            CM_proj[i,j] = compute_cm_HJB(pss_proj_base,tol=tol_HJB)
            if return_intrinsic_cm:
                CM_intr[i,j] = compute_cm_HJB(pss_intr_base,tol = tol_HJB)

    if return_intrinsic_cm:
        return V_theta, V_phi, CM_proj, X, pss_ambient, CM_intr
    else :
        return V_theta, V_phi, CM_proj, X, pss_ambient
    



def cm_th(n,psstype = 1):
    """Compute the theoretical cosine measure of an ambiant PSS of given type in R^n."""
    if psstype == 1:
        cm = 1/np.sqrt(n)
    elif psstype == 2:
        cm = 1/np.sqrt(n**2+2*(n-1)*np.sqrt(n))
    elif psstype == 3:
        cm = 1/n
    else : 
        raise ValueError("psstype must be 1, 2 or 3")
    return cm
    
        

def CM_compute_anydim(m,psstype = 1,n_samples = 30,part_points = [], testing_intr_pss = False,tol_manifold = 1e-12, tol_HJB = 10**(-10), tol = 10**(-10)):
    """
    Args :
        m (int): m>2. Sphere manifold dimension.
        psstype (int): 1 or 2 or 3. type of the pss we want to project (same order as table 1 in the paper).
        n_samples (int): number of sample points on the sphere.
        part_points (list): list of k vectors on the sphere on which one wants to compute the cm of the projection. empty list by default.
        testing_intr_pss (boolean): Parameter that checks that the cosine measure algorithm is coherent with theoretical values of intrinsic pss.
        tol_manifold : tolerance before raising "out of manifold" error.
        tol_HJB : tolerance to decide if the det of a family of vectors is !=0 or not
        tol : tolerance for the renormalization of vectors
    
    Returns :
        X (list): array of sampling vectors on the sphre
        part_points (list) : list of particular points on which the cm was computed
        CM (list): cosine measure of the projected pss on the tangent space at points x in X.
        CM_part_points (list): same but for x in part_points
        cm_intr_th (float): theoretical value of the cosine measure of the intrinsic pss of type psstype
        cm_ambiant_th (float): theoretical value of the cosine measure of the ambiant pss of type psstype
        CM_intr (list) : if testing_intr_pss = True, list of cosine measures of the intrinsic pss (must contains constant values).
    """
    n = m + 1
    nbr_part_points = len(part_points) # index to slice the result for particular points.
    for part_point in part_points:
        if np.abs(np.linalg.norm(part_point, ord = 2)- 1) > tol_manifold:
            raise ValueError("object not in sphere. check if this is due to implementation or numerical error")

    sphere = pymanopt.manifolds.Sphere(n)
    points = part_points + [sphere.random_point() for _ in range(n_samples)]

   
    pss = generate_PSS(n,psstype = psstype)
    CM = []
    CM_intr = []
    print("################ Entering dimension n =  {:}".format(n))

    for k in tqdm(range(len(points))):
        point = points[k]

        # Building base of tangent space
        matrix = np.random.randn(n,n)
        matrix[:,0] = point 
        Q,_ = lg.qr(matrix)
        base_tangentspace = Q[:,1:] # basis of n-1 vecteurs if T_x M (in columns)
        
        # building intrinsic pss in T_x M 
        pss_intr = base_tangentspace.dot(generate_PSS(m,psstype=psstype))

        # building and normalizing projected pss 
        pss_proj = np.array([sphere.to_tangent_space(point, vec) for vec in pss.T]).T
        normalized_vectors,indices_zero, indices_keep = normalize(pss_proj, tol = tol)
        pss_proj = normalized_vectors[:,indices_keep]

        # decomposing vectors of pssproj in this basis : matrix with m lines.
        pss_proj_base_tangentspace = base_tangentspace.T.dot(pss_proj)
        pss_intr_base_tangentspace = base_tangentspace.T.dot(pss_intr) # reverse operation (to check the implementation).


        # Running the cosine measure computation loop
        CM.append(compute_cm_HJB(pss_proj_base_tangentspace, tol = tol_HJB))


        if testing_intr_pss: # testing that I get the good computation for "trivial pss"
            CM_intr.append(compute_cm_HJB(pss_intr_base_tangentspace, tol = tol_HJB))
    
    CM = np.array(CM)
    CM_part_points = CM[:nbr_part_points]
    CM = CM[nbr_part_points:]
    X = points[nbr_part_points:]
    cm_intr_th = cm_th(n = m, psstype = psstype)
    cm_ambiant_th = cm_th(n = n, psstype = psstype)
    return X, part_points, CM, CM_part_points, cm_intr_th, cm_ambiant_th, CM_intr



def upper_bound_projcm(n,psstype=1, attempt_tight = False):
    """compute an upper bound on the projected cosine measure of type 1 fucntion over the sphere"""
    if psstype != 1:
        raise NotImplementedError("not implemented yet!")
    else:
        if attempt_tight == False:
            value = 1/np.sqrt(n-2+1/n)
        else: 
            if n%2 == 1:
                value = 1/np.sqrt(n-2+1/n)
            else:
                value = 1/np.sqrt(n-2+1/(n-1)) 
    return value

def lower_bound_projcm(n,psstype=1):
    """compute a lower bound on the projected cosine measure of type 1 fucntion over the sphere"""
    if psstype != 1:
        raise NotImplementedError("not implemented yet!")
    else:
        value = 1/np.sqrt(n-1)
    return value



def compute_cm_with_dims(adims, psstype= 1,n_samples = 100, with_part_points = True):
    """
    Compute the cosine measure of projected PSS at various points on spheres of various dimensions.
    Args :
        adims (list): list of ambient dimensions (n) for which we want to compute the cosine measure.
        psstype (int): 1 or 2 or 3. type of the pss we want to project (same order as table 1 in the paper).
        n_samples (int): number of random sample points on each sphere.
        with_part_points (boolean): add specific points (see implementation for details).
    Returns :
        CMS_mean (ndarray): mean value of the cosine measure over the sampled points for each dimension.
        CMS_sd (ndarray): standard deviation of the cosine measure over the sampled points for each dimension.
        CMS_min (ndarray): minimum value of the cosine measure over the sampled points for each dimension.
        CMS_max (ndarray): maximum value of the cosine measure over the sampled points for each dimension.
        CMS_part_points (ndarray): cosine measure values at the specific points for each dimension.
        CMS_intr (ndarray): theoretical intrinsic cosine measure for each dimension.
        part_points_labels (list): labels of the specific points.
        part_points_colors (list): colors for plotting the specific points.
    """
    CMS = np.zeros((n_samples,len(adims)))
    nb_part_points = 3*(psstype ==1) + 2*(psstype ==2) + 1*(psstype == 3)
    CMS_part_points = np.zeros((nb_part_points,len(adims)))
    CMS_intr = np.zeros(len(adims))
    for k,n in enumerate(adims):

        m = n-1 
        # construction des points particuliers, intuités par la sphere
        if with_part_points:
            if psstype == 1:
                part_points = np.tri(n,n,0).T/np.sqrt(np.arange(1,n+1))
                part_points = part_points[:,[0,-2,-1]]
                part_points = [part_point for part_point in part_points.T]
                part_points_labels = ["$x$: $k=1$", "$x$: $k=n-1$", "$x$: $k=n$"]
                part_points_colors = ["red","green","purple"]
            elif psstype == 2:
                eye = np.eye(n)
                part_point = (eye[:,0]+eye[:,1])/np.sqrt(2)
                part_point_bis = np.mean(eye[:,:-1],axis = 1)
                part_point_bis = part_point_bis/np.linalg.norm(part_point_bis,ord = 2)
                part_points = [part_point,part_point_bis]
                part_points_labels = ["x=mean_(e1,e2)", "x=mean (e1, ..,e_{n-1})"]
                print(part_points)
                part_points_colors = ["orange","red"]
            elif psstype == 3:
                pss = generate_PSS(n,3)
                part_point = (pss[:,0]+pss[:,1])/2
                part_point = part_point/np.linalg.norm(part_point)
                part_points = [part_point]
                part_points_labels = ["x=mean_(e1,e2)"]
                part_points_colors = ["orange"]

        else :
            part_points = []
        
        X, part_points, CM, CM_part_points, cm_intr_th, cm_ambiant_th, CM_intr= CM_compute_anydim(m,psstype = psstype,n_samples = n_samples,part_points = part_points, testing_intr_pss = False,tol_manifold = 1e-15 , tol_HJB = 10**(-10), tol = 10**(-10))

        CMS[:,k] = CM
        CMS_part_points[:,k] = CM_part_points 
        CMS_intr[k] = cm_th(m,psstype = psstype)

    CMS_mean = np.mean(CMS, axis = 0)
    CMS_max = np.max(CMS, axis = 0) 
    CMS_min = np.min(CMS, axis = 0)
    CMS_sd = np.sqrt(np.mean((CMS-CMS_mean)**2,axis = 0))
    return CMS_mean,CMS_sd, CMS_min, CMS_max, CMS_part_points, CMS_intr,part_points_labels, part_points_colors
    

