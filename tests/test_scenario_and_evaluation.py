"""Vehikel (Reproduzierbarkeit, bitidentisch zu nsga2-demo/genetic-algorithm-demo) und Auswertung (Objektive,
Brute-Force-Front, Sweep, beide Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import moead_algorithm as A
import moead_constants as C
import moead_evaluation as E
import moead_scenario as S


def test_generate_perm_is_reproducible_and_shaped():
    a = S.generate_perm(20, cluster_share=30, seed=7)
    b = S.generate_perm(20, cluster_share=30, seed=7)
    assert np.array_equal(a.xy, b.xy) and np.array_equal(a.co2_factor_matrix, b.co2_factor_matrix)
    assert a.xy.shape == (21, 2) and a.co2_factor_matrix.shape == (21, 21)


def test_generate_perm_matches_nsga2_demo_bit_for_bit_on_the_comparison_instance():
    """Die kleine Vergleichsinstanz (n=8, Seed 19) muss xy/CO2 bitidentisch zu nsga2-demo/genetic-algorithm-demo liefern -
    reproduziert deren generate_perm hier lokal (kein Cross-Repo-Import, wie überall im Portfolio)."""
    def ref_generate_perm(n, cluster_share, seed):
        rng = np.random.default_rng(seed)
        n_grouped = int(round(n * cluster_share / 100))
        uniform = rng.random((n - n_grouped, 2)) * C.AREA
        centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
        which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
        grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
        depot = np.array([[C.AREA / 2, C.AREA / 2]])
        xy = np.vstack([depot, uniform, grouped])
        n_nodes = n + 1
        raw = rng.uniform(C.CO2_FACTOR_LO, C.CO2_FACTOR_HI, size=(n_nodes, n_nodes))
        co2 = (raw + raw.T) / 2.0
        np.fill_diagonal(co2, 0.0)
        return xy, co2

    xy_ref, co2_ref = ref_generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    inst = S.generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    assert np.array_equal(inst.xy, xy_ref)
    assert np.array_equal(inst.co2_factor_matrix, co2_ref)


def test_objective_fn_has_two_objectives_and_matches_manual_computation():
    s = E.Settings(n=5, seed=1, pop=10, gens=5)
    fn, n_nodes = E.objective_fn(s)
    pop = np.array([np.arange(n_nodes)])
    obj = fn(pop)
    assert obj.shape == (1, 2)
    inst, D = E.instance(s.n, s.cluster_share, s.seed)
    dist = A.tour_length_batch(pop, D)
    co2 = A.tour_edge_cost_batch(pop, D, inst.co2_factor_matrix)
    assert obj[0, 0] == pytest.approx(dist[0]) and obj[0, 1] == pytest.approx(co2[0])


def test_brute_force_front_is_non_dominated_and_deduplicated_at_small_n():
    s = E.Settings(n=6, seed=3)
    fn, n_nodes = E.objective_fn(s)
    all_obj, front = E.brute_force_front(n_nodes, fn)
    assert len(all_obj) == 720
    assert len(np.unique(front, axis=0)) == len(front)
    assert not A.dominance_matrix(front).any()


def test_comparison_instance_front_size_matches_genetic_algorithm_demos_measurement():
    s = E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED)
    fn, n_nodes = E.objective_fn(s)
    _, front = E.brute_force_front(n_nodes, fn)
    assert len(front) == C.GA_WEIGHTED_SUM_FRONT_SIZE == 8


def test_run_config_and_sweep_smoke():
    rows = E.sweep("pop", base=E.Settings(n=10, gens=20), values=(20, 30))
    assert len(rows) == 2
    assert all(r["archive_size"] >= 0 for r in rows)


def test_comparison_experiment_smoke_small():
    report = E.comparison_experiment(n=6, seed=3, pop=20, gens=30, seeds=(1, 2))
    assert report["front_size"] >= 1
    assert 0 <= report["reached_median"] <= report["front_size"]
    assert report["ga_weighted_sum_reached"] == C.GA_WEIGHTED_SUM_REACHED
    assert report["nsga2_reached"] == C.NSGA2_REACHED


def test_neighborhood_experiment_smoke_small():
    rows = E.neighborhood_experiment(n=10, pop=20, gens=20, values=(3, 6), seeds=(1, 2))
    assert len(rows) == 2
    assert all(r["archive_size_median"] >= 0 for r in rows)
