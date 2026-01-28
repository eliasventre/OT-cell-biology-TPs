import numpy as np
from scipy.spatial.distance import cdist
import ot

def compute_ot_coupling_manual(X_source, X_target, epsilon, max_iter=1000, tol=1e-6):
    """
    Implémenter l'algorithme de Sinkhorn à la main.
    
    Parameters
    ----------
    X_source : ndarray, shape (n_source, d)
        Particules sources
    X_target : ndarray, shape (n_target, d)
        Particules cibles
    epsilon : float
        Paramètre de régularisation entropique
        
    Returns
    -------
    gamma : ndarray, shape (n_source, n_target)
        Plan de transport optimal (matrice de couplage)
    """
    
    return np.ones((X_source.shape[0], X_target.shape[0]))


def compute_ot_coupling(X_source, X_target, epsilon):
    """
    Implémenter Sinkhorn avec ot.sinkhorn.
    
    Parameters
    ----------
    X_source : ndarray, shape (n_source, d)
        Particules sources
    X_target : ndarray, shape (n_target, d)
        Particules cibles
    epsilon : float
        Paramètre de régularisation entropique
    
    Returns
    -------
    gamma : ndarray, shape (n_source, n_target)
        Plan de transport optimal
    """

    return np.ones((X_source.shape[0], X_target.shape[0]))


def build_trajectories(snapshots_dict, couplings, n_trajectories=100, seed=42):
    """
    Construit des trajectoires en chaînant les couplages OT.
    
    Parameters
    ----------
    snapshots_dict : dict
        Dictionnaire {temps: array de particules}
    couplings : dict
        Dictionnaire {(t_start, t_end): matrice de couplage}
    n_trajectories : int
        Nombre de trajectoires à construire
    
    Returns
    -------
    trajectories : list of list of ndarray
        Liste de trajectoires, chaque trajectoire est une liste de positions
    """
    rng = np.random.default_rng(seed)
    times = sorted(snapshots_dict.keys())
    
    # Indices de départ aléatoires
    n_particles_start = len(snapshots_dict[times[0]])
    start_indices = rng.choice(n_particles_start, size=n_trajectories, replace=True)
    
    trajectories = []
    
    for start_idx in start_indices:
        trajectory = []

        for k in range(len(times)):
            t_current = times[k]
            trajectory.append(snapshots_dict[t_current][0])
        
        trajectories.append(trajectory)
    
    return trajectories, times


def mccann_interpolation(X_source, X_target, gamma, t, n_samples=1000, seed=42):
    """
    Calcule l'interpolation de McCann au temps t entre deux distributions.
    
    Parameters
    ----------
    X_source : ndarray, shape (n_source, d)
        Particules sources (au temps 0)
    X_target : ndarray, shape (n_target, d)
        Particules cibles (au temps 1)
    gamma : ndarray, shape (n_source, n_target)
        Plan de transport optimal
    t : float
        Temps d'interpolation (entre 0 et 1)
    n_samples : int
        Nombre d'échantillons à générer
    
    Returns
    -------
    X_interp : ndarray, shape (n_samples, d)
        Particules interpolées au temps t
    """
    
    return (X_source + X_target) / 2


def distribution_distance(X, Y, epsilon=0):
    """
    Calcule la distance de Wasserstein (si epsilon = 0) ou la divergence entropique (si epsilon > 0) entre deux distributions empiriques.
    
    Utilise l'OT exact si epsilon = 0 ou la sinkhorn divergence.
    """

    return 1