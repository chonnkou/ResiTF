import numpy as np
from scipy.sparse.linalg import eigs
import random
import math
import time
import hypernetx as hnx
import generation as gen

def algebraic_connectivity(X):
    """
    Calculate the algebraic connectivity of a binary hypergraph
    X : numpy.ndarray
        Incidence matrix with shape (N, K).
        Rows = nodes (agents)
        Columns = hyperedges (tasks)
        X[i, k] = 1 if node i belongs to hyperedge k.
    Returns
    mu2 : float
        Second-smallest eigenvalue of the EDVW Laplacian.
    """

    X = np.asarray(X, dtype=float)
    N, T = X.shape
    
    # incidence weight or a.k.a. EDVW
    gamma = X.copy()
    # hyperedge weight
    omega = np.sum(X, axis=0)

    # probability transition matrix of the random walk

    # incidence info and node weight
    R = X * gamma
    # incidence info and hyperedge weight
    W = X * omega
    
    # node degs - hyperedge weight W - sum over row - axis=1
    delta = np.sum(W, axis=1)
    # hyperedge degs - node weight R - sum over col - axis=0
    d = np.sum(R, axis=0)
    # make it float
    delta = np.maximum(delta, np.ones(N))
    d = np.maximum(d, np.ones(T))
    # probability transition matrix
    P = (
        np.diag(delta ** -1)
        @ W
        @ np.diag(d ** -1)
        @ R.T
    )

    P = np.nan_to_num(P)

    # construct Pi
    eigenValues, eigenVectors = eigs(
        P.T,
        k=1,
        which='LR'
    )
    pi = np.abs(eigenVectors[:, 0])
    Pi = np.diag(pi)

    # Laplacian
    L = Pi - (Pi @ P + P.T @ Pi) / 2.0

    # Remove tiny numerical asymmetry / imaginary parts
    L = np.real(L)
    L = (L + L.T) / 2.0

    # algebraic connectivity = second-smallest eigenvalue

    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)

    mu2 = eigenvalues[1]

    if np.isclose(mu2, 0.0, atol=1e-12):
        mu2 = 0.0

    return mu2

def X2H(X):
    return hnx.Hypergraph.from_incidence_matrix(X)

def penalty(X, B, R):
    """
    Penalty function:

    f(X) =
        sum_i (sum_j X_ij - B)^+
        +
        sum_j (R - sum_i X_ij)^+

    Parameters
    ----------
    X : np.ndarray
        Binary assignment matrix, shape = (N, T)
    B : int
        Maximum number of tasks assigned to each agent
    R : int
        Minimum number of agents required by each task

    Returns
    -------
    float
        Total constraint violation.
    """
    row_sum = np.sum(X, axis=1)
    col_sum = np.sum(X, axis=0)

    row_penalty = np.maximum(row_sum - B, 0).sum()
    col_penalty = np.maximum(R - col_sum, 0).sum()
    
    # algebraic connectivity
    ac = algebraic_connectivity(X)

    return row_penalty + col_penalty - ac


def generate_initial_solution(N, T, B, R):
    """
    Generate an initial binary assignment matrix.

    The initial density is selected between:
        R / N
    and
        B / T

    so that the expected row and column sums are close to
    the feasible region.
    """

    # Necessary feasibility checks
    if R > N:
        raise ValueError(
            "Infeasible problem: R cannot exceed the number of agents N."
        )

    if B > T:
        B = T

    if N * B < T * R:
        raise ValueError(
            "Infeasible problem: N * B < T * R."
        )

    # Lower density required by column requirement
    p_lower = R / N

    # Upper density allowed by row capacity
    p_upper = B / T

    # Use the middle of the feasible density interval
    p = (p_lower + p_upper) / 2

    X = (np.random.rand(N, T) < p).astype(int)

    return X


def generate_neighbor(X, B, R, random_move_prob=0.10):
    """
    Generate a neighbouring solution using a structure-guided perturbation.

    Main perturbations:
    1. Repair a deficient column.
    2. Repair an overloaded row.
    3. Random flip with a small probability.
    """

    Y = X.copy()

    N, T = Y.shape

    row_sum = np.sum(Y, axis=1)
    col_sum = np.sum(Y, axis=0)

    # ---------------------------------------------------------
    # Random perturbation
    # ---------------------------------------------------------
    if random.random() < random_move_prob:

        i = random.randrange(N)
        j = random.randrange(T)

        Y[i, j] = 1 - Y[i, j]

        return Y

    # ---------------------------------------------------------
    # Identify violations
    # ---------------------------------------------------------

    overloaded_rows = np.where(row_sum > B)[0]
    deficient_cols = np.where(col_sum < R)[0]

    # ---------------------------------------------------------
    # Case 1:
    # There are deficient columns.
    #
    # Add an agent to a deficient task.
    # ---------------------------------------------------------
    if len(deficient_cols) > 0:

        # Prefer columns with larger deficit
        deficits = R - col_sum[deficient_cols]
        max_deficit = np.max(deficits)

        candidates_j = deficient_cols[deficits == max_deficit]

        j = np.random.choice(candidates_j)

        # Rows where X_ij = 0
        candidate_rows = np.where(Y[:, j] == 0)[0]

        if len(candidate_rows) > 0:

            # Prefer rows that are not full
            feasible_rows = candidate_rows[row_sum[candidate_rows] < B]

            if len(feasible_rows) > 0:

                # Prefer the least-loaded rows
                loads = row_sum[feasible_rows]
                min_load = np.min(loads)

                best_rows = feasible_rows[loads == min_load]

                i = np.random.choice(best_rows)

            else:
                # All possible rows are already full.
                # SA may temporarily generate an infeasible row.
                loads = row_sum[candidate_rows]
                min_load = np.min(loads)

                best_rows = candidate_rows[loads == min_load]

                i = np.random.choice(best_rows)

            Y[i, j] = 1

            return Y

    # ---------------------------------------------------------
    # Case 2:
    # There are overloaded rows.
    #
    # Remove one assignment from an overloaded agent.
    # ---------------------------------------------------------
    if len(overloaded_rows) > 0:

        excess = row_sum[overloaded_rows] - B
        max_excess = np.max(excess)

        candidate_i = overloaded_rows[excess == max_excess]

        i = np.random.choice(candidate_i)

        # Tasks currently assigned to agent i
        assigned_cols = np.where(Y[i, :] == 1)[0]

        if len(assigned_cols) > 0:

            # Prefer removing from columns having surplus agents
            surplus_cols = assigned_cols[col_sum[assigned_cols] > R]

            if len(surplus_cols) > 0:

                surplus = col_sum[surplus_cols] - R

                max_surplus = np.max(surplus)

                best_cols = surplus_cols[surplus == max_surplus]

                j = np.random.choice(best_cols)

            else:
                # No column has surplus.
                # Allow SA to temporarily make a column deficient.
                j = np.random.choice(assigned_cols)

            Y[i, j] = 0

            return Y

    # ---------------------------------------------------------
    # Case 3:
    # No violation detected.
    #
    # This normally means penalty = 0.
    # Generate a random neighbour if search continues.
    # ---------------------------------------------------------

    i = random.randrange(N)
    j = random.randrange(T)

    Y[i, j] = 1 - Y[i, j]

    return Y
    
def time_used(t):
    tc = f'{t} secs' if t < 60 else f'{t / 60} mins'
    return tc


def simulated_annealing(
        N,
        T,
        B,
        R,
        initial_temperature=100.0,
        cooling_rate=0.995,
        minimum_temperature=1e-4,
        max_time=60,
        random_move_prob=0.5,
        seed=None
):
    """
    Simulated annealing for binary assignment matrix generation.

    Parameters
    ----------
    N : int
        Number of agents.
    T : int
        Number of tasks.
    B : int
        Maximum number of tasks assigned to each agent.
    R : int
        Minimum number of agents assigned to each task.
    initial_temperature : float
        Initial SA temperature.
    cooling_rate : float
        Geometric cooling coefficient.
    minimum_temperature : float
        Minimum temperature.
    max_iterations : int
        Maximum number of iterations.
    random_move_prob : float
        Probability of performing a completely random flip.
    seed : int or None
        Random seed.

    Returns
    -------
    best_X : np.ndarray
        Best binary assignment matrix.
    best_penalty : float
        Penalty of the best solution.
    history : list
        Best penalty during search.
    """
    start_t = time.time()
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # ---------------------------------------------------------
    # Feasibility check
    # ---------------------------------------------------------

    if N * B < T * R:
        raise ValueError(
            f"Infeasible parameters: N*B = {N * B} < "
            f"T*R = {T * R}."
        )

    # ---------------------------------------------------------
    # Initial solution
    # ---------------------------------------------------------

    current_X = generate_initial_solution(
        N=N,
        T=T,
        B=B,
        R=R
    )

    current_penalty = penalty(
        current_X,
        B,
        R
    )

    best_X = current_X.copy()
    best_penalty = current_penalty

    temperature = initial_temperature

    history = [best_penalty]

    # ---------------------------------------------------------
    # Simulated annealing
    # ---------------------------------------------------------

#    for iteration in range(max_iterations):
    while True:
        current_t = time.time()
        if current_t - start_t >= max_time:
            break
        # Stop immediately once a zero-penalty solution is found
        if best_penalty == 0:
            break

        # Generate neighbour
        candidate_X = generate_neighbor(
            current_X,
            B,
            R,
            random_move_prob=random_move_prob
        )

        candidate_penalty = penalty(
            candidate_X,
            B,
            R
        )

        delta = candidate_penalty - current_penalty

        # -----------------------------------------------------
        # Metropolis acceptance criterion
        # -----------------------------------------------------

        if delta <= 0:

            accept = True

        else:

            acceptance_probability = math.exp(
                -delta / temperature
            )

            accept = random.random() < acceptance_probability

        # -----------------------------------------------------
        # Accept candidate
        # -----------------------------------------------------

        if accept:

            current_X = candidate_X
            current_penalty = candidate_penalty

        # -----------------------------------------------------
        # Update global best
        # -----------------------------------------------------

        if current_penalty < best_penalty:

            best_X = current_X.copy()
            best_penalty = current_penalty

        history.append(best_penalty)
        

        # -----------------------------------------------------
        # Cooling
        # -----------------------------------------------------

        temperature *= cooling_rate

        # Optional lower bound on temperature
        if temperature < minimum_temperature:
            temperature = minimum_temperature
        
    end_t = time.time()
    usage_t = end_t - start_t
    return X2H(best_X)
#    return X2H(best_X), best_penalty, history, time_used(usage_t)