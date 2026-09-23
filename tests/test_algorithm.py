"""Kern gegen Handrechnung + Bibliotheksgegenprobe (pymoo). Der kopierte Dominanz-/Operatoren-Kern wird knapp erneut
geprüft (volle Kreuzprobe steht in nsga2-demo); die neue MOEA/D-Mechanik (Gewichtsvektoren, Nachbarschaften,
Tchebycheff-Skalarisierung, Archiv, Hauptschleife) wird vollständig geprüft - Tchebycheff stimmt exakt mit
`pymoo.decomposition.tchebicheff.Tchebicheff` überein."""

from itertools import permutations

import numpy as np
import pytest
from pymoo.decomposition.tchebicheff import Tchebicheff

import moead_algorithm as A

# --- Kopierter Kern: knappe erneute Kreuzprobe (volle Prüfung steht in nsga2-demo) -----------------------------------------------------------


def test_order_crossover_matches_hand_calculation():
    class _FixedRNG:
        def integers(self, lo, hi, size=None):
            return np.array([3, 6])
    p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    p2 = np.array([5, 4, 6, 9, 2, 1, 7, 8, 3])
    assert A.order_crossover(p1, p2, _FixedRNG()).tolist() == [9, 2, 1, 4, 5, 6, 7, 8, 3]


def test_non_dominated_mask_matches_hand_example():
    points = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0], [2.0, 5.0], [3.0, 3.0], [5.0, 2.0]])
    mask = A.non_dominated_mask(points)
    assert sorted(np.where(mask)[0].tolist()) == [0, 1, 2]


# --- Gewichtsvektoren / Nachbarschaften -----------------------------------------------------------------------------------------------------


def test_generate_weight_vectors_matches_hand_calculation():
    w = A.generate_weight_vectors(5)
    expected = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
    np.testing.assert_allclose(w, expected)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_neighborhoods_matches_hand_calculation():
    w = A.generate_weight_vectors(5)      # [0,1],[.25,.75],[.5,.5],[.75,.25],[1,0] - gleichmäßig auf einer Geraden
    neigh = A.neighborhoods(w, t=3)
    # Für Punkt 2 (Mitte, w=[.5,.5]) sind die beiden Nachbarn 1 und 3 am nächsten (gleicher Abstand), dann sich selbst
    assert set(neigh[2].tolist()) == {1, 2, 3}
    # Für Punkt 0 (Rand) sind die T=3 nächsten 0, 1, 2 (aufsteigender Abstand entlang der Geraden)
    assert neigh[0].tolist() == [0, 1, 2]


def test_neighborhoods_always_includes_self_first():
    rng = np.random.default_rng(0)
    w = A.generate_weight_vectors(20)
    neigh = A.neighborhoods(w, t=5)
    for i in range(20):
        assert i in neigh[i].tolist()


# --- Tchebycheff: exakte Kreuzprobe gegen pymoo + Handrechnung -----------------------------------------------------------------------------


def test_tchebycheff_matches_hand_calculation():
    F = np.array([3.0, 4.0])
    w = np.array([0.5, 0.5])
    ideal = np.array([1.0, 1.0])
    # |3-1|*0.5=1.0, |4-1|*0.5=1.5 -> max = 1.5
    assert A.tchebycheff(F, w, ideal)[0] == pytest.approx(1.5)


@pytest.mark.parametrize("seed", range(15))
def test_tchebycheff_matches_pymoo(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 20))
    F = rng.random((n, 2)) * 100
    w = rng.random((n, 2))
    w = w / w.sum(axis=1, keepdims=True)
    ideal = F.min(axis=0) - rng.random(2) * 5     # Idealpunkt nicht zwingend im Datensatz enthalten
    mine = A.tchebycheff(F, w, ideal)
    pym = Tchebicheff().do(F, w, ideal_point=ideal, _type="one_to_one")
    np.testing.assert_allclose(mine, pym, rtol=1e-9)


# --- Externes Archiv -----------------------------------------------------------------------------------------------------------------------


def test_update_archive_keeps_only_non_dominated_and_deduplicates():
    archive = np.array([[1.0, 5.0], [2.0, 3.0]])
    new = np.array([[4.0, 1.0], [2.0, 3.0], [3.0, 3.0]])    # (3,3) dominiert von (2,3); (2,3) ist ein Duplikat
    updated = A.update_archive(archive, new)
    assert sorted(updated.tolist()) == [[1.0, 5.0], [2.0, 3.0], [4.0, 1.0]]


def test_update_archive_removes_points_dominated_by_new_ones():
    archive = np.array([[5.0, 5.0]])
    new = np.array([[3.0, 3.0]])            # dominiert [5,5] eindeutig
    updated = A.update_archive(archive, new)
    assert updated.tolist() == [[3.0, 3.0]]


def test_update_archive_from_empty():
    updated = A.update_archive(np.empty((0, 2)), np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert len(updated) == 2


# --- MOEA/D-Lauf auf einer sehr kleinen Instanz: nachweislich die Mehrheit der Brute-Force-Front (Mehrheit, nicht jeder Seed) ----------------


def _brute_force_front(n_nodes, fn):
    tours = np.array([(0,) + p for p in permutations(range(1, n_nodes))])
    obj = fn(tours)
    unique_obj = np.unique(obj, axis=0)
    return unique_obj[A.non_dominated_mask(unique_obj)]


def test_moead_finds_most_of_the_brute_force_front_on_a_tiny_instance():
    rng = np.random.default_rng(42)
    n_nodes = 6
    xy = rng.random((n_nodes, 2)) * 100.0
    D = A.dist_matrix(xy)
    raw = rng.uniform(0.6, 3.4, size=(n_nodes, n_nodes))
    factor = (raw + raw.T) / 2.0
    np.fill_diagonal(factor, 0.0)

    def fn(pop):
        dist = A.tour_length_batch(pop, D)
        co2 = A.tour_edge_cost_batch(pop, D, factor)
        return np.stack([dist, co2], axis=1)

    true_front = _brute_force_front(n_nodes, fn)
    hits = []
    for seed in range(15):
        r = A.run_moead(fn, n_nodes, pop_size=20, generations=60, cx_prob=0.9, mut_prob=0.2, t=5, neighbor_prob=0.9, update_limit=2, seed=seed)
        reached = sum(np.any(np.all(np.abs(r.archive - p) < 1e-6, axis=1)) for p in true_front)
        hits.append(reached / len(true_front))
    assert np.median(hits) >= 0.6


def test_run_moead_result_shapes_and_archive_history_length():
    rng = np.random.default_rng(0)
    n_nodes = 10
    xy = rng.random((n_nodes, 2)) * 100.0
    D = A.dist_matrix(xy)
    raw = rng.uniform(0.6, 3.4, size=(n_nodes, n_nodes))
    factor = (raw + raw.T) / 2.0
    np.fill_diagonal(factor, 0.0)

    def fn(pop):
        return np.stack([A.tour_length_batch(pop, D), A.tour_edge_cost_batch(pop, D, factor)], axis=1)

    r = A.run_moead(fn, n_nodes, pop_size=20, generations=15, cx_prob=0.9, mut_prob=0.2, t=5, neighbor_prob=0.9, update_limit=2, seed=1, keep_history=True)
    assert r.final_population.shape == (20, n_nodes)
    assert r.final_objectives.shape == (20, 2)
    assert len(r.archive_history) == 16       # 0 (Start) + 15 Generationen
    assert len(r.generations) == 16
    assert not A.dominance_matrix(r.archive).any()      # Archiv ist selbst nicht-dominiert
