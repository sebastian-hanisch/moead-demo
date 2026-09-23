"""MOEA/D (Zhang & Li, 2007), numpy von Grund auf. Order Crossover, Tausch-Mutation und die Distanz-/Dominanz-Hilfen sind
wortgleich aus nsga2-demo/nsga2_algorithm.py kopiert; neu ist die komplette Selektions-/Ersetzungsmechanik - **Zerlegung**
statt Dominanz-Sortierung: das Mehrzielproblem wird in `pop_size` skalare Unterprobleme zerlegt (ein Gewichtsvektor je
Unterproblem, Tchebycheff-Skalarisierung), jedes Unterproblem paart und ersetzt nur innerhalb seiner **Nachbarschaft**
(die T Unterprobleme mit dem ähnlichsten Gewichtsvektor) statt über die ganze Population. Ein externes Archiv sammelt die
nicht-dominierten Zielwerte über den GANZEN Lauf (nicht nur die aktuelle Population) - MOEA/Ds Arbeitspopulation kann gute
Lösungen wieder verlieren, wenn ihr Unterproblem lokal überschrieben wird."""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


# ===============================================================================================================================
# Kopiert aus nsga2-demo/nsga2_algorithm.py (Distanz/Kantenkosten, Operatoren, Dominanz-Hilfen für das externe Archiv)
# ===============================================================================================================================


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def tour_length_batch(pop, D):
    nxt = np.roll(pop, -1, axis=1)
    return D[pop, nxt].sum(axis=1)


def tour_edge_cost_batch(pop, D, factor_matrix):
    nxt = np.roll(pop, -1, axis=1)
    return (D[pop, nxt] * factor_matrix[pop, nxt]).sum(axis=1)


def tour_edges(tour):
    t = np.asarray(tour)
    a, b = t, np.roll(t, -1)
    return {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b)}


def edge_share(tour, reference):
    return len(tour_edges(tour) & tour_edges(reference)) / len(tour)


def init_population(pop_size, n_nodes, rng):
    return np.array([rng.permutation(n_nodes) for _ in range(pop_size)], dtype=np.int64)


def order_crossover(p1, p2, rng):
    n = len(p1)
    i, j = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=np.int64)
    child[i:j + 1] = p1[i:j + 1]
    taken = set(child[i:j + 1].tolist())
    fill = [g for g in p2.tolist() if g not in taken]
    pos = [k for k in range(n) if not (i <= k <= j)]
    for k, g in zip(pos, fill):
        child[k] = g
    return child


def swap_mutation(ind, p_mut, rng):
    if rng.random() >= p_mut:
        return ind.copy()
    out = ind.copy()
    i, j = rng.integers(0, len(ind), size=2)
    out[i], out[j] = out[j], out[i]
    return out


def dominance_matrix(objectives):
    F = np.asarray(objectives, dtype=float)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    return dom


def non_dominated_mask(objectives):
    """Boolmaske der nicht-dominierten Punkte ohne volle (N, N)-Matrix - O(N * F) statt O(N²), siehe nsga2-demo."""
    F = np.asarray(objectives, dtype=float)
    order = np.argsort(F[:, 0])
    frontier_idx = []
    frontier_obj = np.empty((0, F.shape[1]))
    for idx in order:
        p = F[idx]
        if len(frontier_obj) and np.any(np.all(frontier_obj <= p, axis=1) & np.any(frontier_obj < p, axis=1)):
            continue
        if len(frontier_obj):
            dominated_by_p = np.all(p <= frontier_obj, axis=1) & np.any(p < frontier_obj, axis=1)
            keep = ~dominated_by_p
            frontier_idx = [fi for fi, k in zip(frontier_idx, keep.tolist()) if k]
            frontier_obj = frontier_obj[keep]
        frontier_idx.append(int(idx))
        frontier_obj = np.vstack([frontier_obj, p])
    mask = np.zeros(len(F), dtype=bool)
    mask[frontier_idx] = True
    return mask


# ===============================================================================================================================
# MOEA/D (Zhang & Li, 2007) - neu
# ===============================================================================================================================


def generate_weight_vectors(n_points):
    """Gleichmäßig verteilte Gewichtsvektoren auf der Zielraum-Simplex-Kante (2 Ziele): w_i = (i/(N-1), 1 - i/(N-1))."""
    w1 = np.linspace(0.0, 1.0, n_points)
    return np.stack([w1, 1.0 - w1], axis=1)


def neighborhoods(weights, t):
    """Für jeden Gewichtsvektor die Indizes der T nächsten (inklusive sich selbst), nach euklidischem Abstand sortiert."""
    diff = weights[:, None, :] - weights[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return np.argsort(dist, axis=1)[:, :t]


def tchebycheff(F, weights, ideal):
    """Tchebycheff-Skalarisierung: max_k w_k * |f_k(x) - z_k|. F/weights: gleich viele Zeilen (oder beide 1D für einen
    einzelnen Punkt - gibt dann ein Array der Länge 1 zurück)."""
    F = np.atleast_2d(F)
    weights = np.atleast_2d(weights)
    return (np.abs(F - ideal) * weights).max(axis=1)


def update_archive(archive_obj, new_obj):
    """Aktualisiert das externe Archiv (bereits nicht-dominiert, eindeutig) mit neuen Kandidaten - nicht-dominierte,
    eindeutige Zielwerte über die Vereinigung. `archive_obj` kann leer sein (Startaufruf)."""
    combined = np.vstack([archive_obj, new_obj]) if len(archive_obj) else new_obj
    unique_obj = np.unique(combined, axis=0)
    return unique_obj[non_dominated_mask(unique_obj)]


@dataclass
class Generation:
    population: np.ndarray
    objectives: np.ndarray


@dataclass
class MOEADResult:
    final_population: np.ndarray
    final_objectives: np.ndarray
    weights: np.ndarray
    archive: np.ndarray                                       # (a, M) nicht-dominierte Zielwerte über den GANZEN Lauf
    archive_history: list = field(default_factory=list)        # Archiv-Schnappschuss je Generation (0 = Startpopulation)
    generations: list = field(default_factory=list)             # nur bei keep_history=True


def run_moead(objective_fn, n_nodes, pop_size, generations, cx_prob, mut_prob, t, neighbor_prob, update_limit, seed, keep_history=False):
    """objective_fn(population) -> (pop_size, n_obj), niedriger ist besser je Ziel. Nur für 2 Ziele ausgelegt
    (`generate_weight_vectors`), wie das übrige MOEA/D-Kontrast-Stück dieser Linie."""
    rng = np.random.default_rng(seed)
    weights = generate_weight_vectors(pop_size)
    neigh = neighborhoods(weights, t)

    pop = init_population(pop_size, n_nodes, rng)
    obj = objective_fn(pop)
    ideal = obj.min(axis=0)
    archive_obj = update_archive(np.empty((0, obj.shape[1])), obj)
    archive_history = [archive_obj.copy()]
    gens = [Generation(pop.copy(), obj.copy())] if keep_history else []

    for _ in range(generations):
        offspring_obj = []
        for i in rng.permutation(pop_size):
            pool = neigh[i] if rng.random() < neighbor_prob else np.arange(pop_size)
            p_idx = rng.choice(pool, size=2, replace=False)
            p1, p2 = pop[p_idx[0]], pop[p_idx[1]]
            child = order_crossover(p1, p2, rng) if rng.random() < cx_prob else p1.copy()
            child = swap_mutation(child, mut_prob, rng)
            child_obj = objective_fn(child[None, :])[0]
            offspring_obj.append(child_obj)
            ideal = np.minimum(ideal, child_obj)

            n_updated = 0
            for j in rng.permutation(pool):
                if n_updated >= update_limit:
                    break
                if tchebycheff(child_obj, weights[j], ideal)[0] <= tchebycheff(obj[j], weights[j], ideal)[0]:
                    pop[j] = child
                    obj[j] = child_obj
                    n_updated += 1

        archive_obj = update_archive(archive_obj, np.array(offspring_obj))
        archive_history.append(archive_obj.copy())
        if keep_history:
            gens.append(Generation(pop.copy(), obj.copy()))

    return MOEADResult(pop, obj, weights, archive_obj, archive_history, gens)
