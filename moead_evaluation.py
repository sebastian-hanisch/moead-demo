"""Auswertung der MOEA/D-Demo: ein Lauf, Sweep über Populationsgröße/Nachbarschaftsgröße, und zwei Experimente -
Abdeckung der Brute-Force-Pareto-Front im Vergleich zu gewichteter Summe (GA-Demo) und NSGA-II (nsga2-demo) auf der
identischen kleinen Instanz, und die Wirkung der Nachbarschaftsgröße T auf die Archivgröße."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

import numpy as np

import moead_algorithm as A
import moead_constants as C
import moead_scenario as S

BRUTE_FORCE_MAX_N = 9


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = 0
    seed: int = C.DEFAULT_SEED
    pop: int = C.DEFAULT_POP
    gens: int = C.DEFAULT_GEN
    cx: float = C.DEFAULT_CX
    mut: float = C.DEFAULT_MUT
    t: int = C.DEFAULT_T
    neighbor_prob: float = C.DEFAULT_NEIGHBOR_PROB
    update_limit: int = C.DEFAULT_UPDATE_LIMIT
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate_perm(n, cluster_share, seed)
    return inst, A.dist_matrix(inst.xy)


def objective_fn(settings):
    """(Zielfunktion, Knotenzahl) für `settings`. Immer 2 Ziele: Distanz, CO2."""
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)

    def fn(pop):
        dist = A.tour_length_batch(pop, D)
        co2 = A.tour_edge_cost_batch(pop, D, inst.co2_factor_matrix)
        return np.stack([dist, co2], axis=1)
    return fn, inst.n_nodes


def run(settings, keep_history=False):
    fn, n_nodes = objective_fn(settings)
    return A.run_moead(fn, n_nodes, settings.pop, settings.gens, settings.cx, settings.mut, settings.t, settings.neighbor_prob, settings.update_limit, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object


def analyse(settings, keep_history=True):
    return Analysis(settings, run(settings, keep_history=keep_history))


# --- Brute-Force-Referenz (kleine Instanzen) -----------------------------------------------------------------------------------------------


def brute_force_front(n_nodes, fn):
    """Alle (n_nodes - 1)! Touren. Gibt (alle Objektive, eindeutige nicht-dominierte Zielwerte) zurück."""
    tours = np.array([(0,) + p for p in permutations(range(1, n_nodes))], dtype=np.int64)
    obj = fn(tours)
    rounded = np.round(obj, A.OBJ_DECIMALS)          # gespiegelte Touren: Zielwerte nur um Fließkomma-Rauschen verschieden
    _, first = np.unique(rounded, axis=0, return_index=True)
    nd_mask = A.non_dominated_mask(rounded[first])
    return obj, obj[first][nd_mask]


def front_coverage(true_front_obj, found_obj, tol=1e-6):
    reached = 0
    for point in true_front_obj:
        if np.any(np.all(np.abs(found_obj - point) < tol, axis=1)):
            reached += 1
    return reached, len(true_front_obj)


# --- Experiment 1: schließt die Linie ab (gewichtete Summe vs. NSGA-II vs. MOEA/D) -----------------------------------------------------------


def comparison_experiment(n=None, seed=None, pop=None, gens=None, seeds=None):
    n = C.COMPARISON_N if n is None else n
    seed = C.COMPARISON_VEHICLE_SEED if seed is None else seed
    pop = C.COMPARISON_POP if pop is None else pop
    gens = C.COMPARISON_GENS if gens is None else gens
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds

    s0 = Settings(n=n, seed=seed, pop=pop, gens=gens)
    fn, n_nodes = objective_fn(s0)
    _, true_front = brute_force_front(n_nodes, fn)

    reached_list = []
    for run_seed in seeds:
        r = run(replace(s0, run_seed=run_seed), keep_history=False)
        reached, _ = front_coverage(true_front, r.archive)
        reached_list.append(reached)
    return {
        "front_size": len(true_front),
        "reached_median": float(np.median(reached_list)),
        "reached_all": reached_list,
        "ga_weighted_sum_reached": C.GA_WEIGHTED_SUM_REACHED,
        "nsga2_reached": C.NSGA2_REACHED,
    }


# --- Experiment 2: Nachbarschaftsgröße T (eigener Regler) -----------------------------------------------------------------------------------


def neighborhood_experiment(n=None, seed=None, pop=None, gens=None, values=None, seeds=None):
    n = C.DEFAULT_N if n is None else n
    seed = C.DEFAULT_SEED if seed is None else seed
    pop = C.NEIGHBORHOOD_POP if pop is None else pop
    gens = C.NEIGHBORHOOD_GENS if gens is None else gens
    values = C.NEIGHBORHOOD_T_VALUES if values is None else values
    seeds = C.NEIGHBORHOOD_SEEDS if seeds is None else seeds

    rows = []
    for t in values:
        sizes = []
        for run_seed in seeds:
            s = Settings(n=n, seed=seed, pop=pop, gens=gens, t=t, run_seed=run_seed)
            r = run(s, keep_history=False)
            sizes.append(len(r.archive))
        rows.append({"t": t, "archive_size_median": float(np.median(sizes)), "archive_size_all": sizes})
    return rows


# --- Sweep: Populationsgröße/Nachbarschaftsgröße vs. Archivgröße --------------------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    sizes = []
    for run_seed in seeds:
        r = run(replace(s0, run_seed=run_seed), keep_history=False)
        sizes.append(len(r.archive))
    return {"archive_size": float(np.mean(sizes))}


def sweep(param, base=None, values=None):
    base = Settings() if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]
