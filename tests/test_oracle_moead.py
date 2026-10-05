"""Orakel-Tests (unabhängiger Rechenweg): nicht-dominierte Menge/Archiv gegen die O(N²)-Definition, Nachbarschaften gegen
den Index-Abstand der gleichmäßigen Gewichtsvektoren, Tchebycheff per Schleife, Brute-Force-Front gegen Aufzählung,
und ein ganzer MOEA/D-Lauf gegen eine Schleifenfassung mit identischem Zufallsstrom (Auswahl, Ersetzung, Idealpunkt,
Archiv je Generation)."""

import itertools

import numpy as np

import moead_algorithm as A
import moead_evaluation as E


def _dom(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _front(rows):
    pts = {tuple(r) for r in rows}
    return sorted(p for p in pts if not any(_dom(q, p) for q in pts))


def test_non_dominated_mask_and_archive_match_definition():
    rng = np.random.default_rng(1)
    for k in range(60):
        n, m = int(rng.integers(2, 20)), int(rng.integers(2, 4))
        F = rng.integers(0, 5, size=(n, m)).astype(float) if k % 2 else rng.random((n, m))
        assert A.non_dominated_mask(F).tolist() == [not any(_dom(F[j], F[i]) for j in range(n)) for i in range(n)]
        split = int(rng.integers(1, n))
        arch = A.update_archive(A.update_archive(np.empty((0, m)), F[:split]), F[split:])
        assert sorted(map(tuple, arch.tolist())) == _front(F)


def test_weights_neighborhoods_and_tchebycheff_match_definition():
    for n in (3, 20, 33, 60):
        w = A.generate_weight_vectors(n)
        assert np.allclose(w[:, 0], np.arange(n) / (n - 1)) and np.allclose(w.sum(axis=1), 1.0)
        for t in (3, 10, 40):
            nb = A.neighborhoods(w, t)
            for i in range(n):
                thr = sorted(abs(i - j) for j in range(n))[nb.shape[1] - 1]
                assert nb[i, 0] == i and len(set(nb[i].tolist())) == nb.shape[1]
                assert all(abs(i - j) <= thr for j in nb[i])        # Gleichstände links/rechts dürfen beliebig fallen
    rng = np.random.default_rng(2)
    F, W, z = rng.random((8, 3)) * 100, rng.random((8, 3)), rng.random(3) * 50
    ref = [max(W[i][k] * abs(F[i][k] - z[k]) for k in range(3)) for i in range(8)]
    assert np.allclose(A.tchebycheff(F, W, z), ref)


def test_brute_force_front_matches_enumeration():
    for seed in (3, 11):
        s = E.Settings(n=6, seed=seed)
        fn, n_nodes = E.objective_fn(s)
        inst, D = E.instance(s.n, s.cluster_share, s.seed)
        pts = set()
        for p in itertools.permutations(range(1, n_nodes)):
            t = (0,) + p
            d = sum(D[t[i], t[(i + 1) % n_nodes]] for i in range(n_nodes))
            c = sum(D[t[i], t[(i + 1) % n_nodes]] * inst.co2_factor_matrix[t[i], t[(i + 1) % n_nodes]] for i in range(n_nodes))
            pts.add((round(float(d), 9), round(float(c), 9)))
        _, front = E.brute_force_front(n_nodes, fn)
        assert {tuple(np.round(r, 9)) for r in front} == set(_front(pts))


def _ref_ox(p1, p2, i, j):
    rest = iter(g for g in p2 if g not in p1[i:j + 1])
    return [p1[k] if i <= k <= j else next(rest) for k in range(len(p1))]


def _ref_run(fn, n_nodes, n_pop, gens, cx, mut, delta, nr, seed, neigh):
    rng = np.random.default_rng(seed)
    w = [[i / (n_pop - 1), 1 - i / (n_pop - 1)] for i in range(n_pop)]
    pop = [rng.permutation(n_nodes).tolist() for _ in range(n_pop)]
    obj = [list(fn(np.array([p]))[0]) for p in pop]
    z = [min(o[m] for o in obj) for m in range(2)]
    hist = [_front(obj)]
    arch = hist[0]

    def tch(f, wj):
        return max(wj[m] * abs(f[m] - z[m]) for m in range(2))

    for _ in range(gens):
        offs = []
        for i in rng.permutation(n_pop):
            pool = list(neigh[i]) if rng.random() < delta else list(range(n_pop))
            a, b = rng.choice(np.array(pool), size=2, replace=False)
            p1, p2 = pop[int(a)], pop[int(b)]
            if rng.random() < cx:
                lo, hi = sorted(rng.integers(0, n_nodes, size=2))
                child = _ref_ox(p1, p2, int(lo), int(hi))
            else:
                child = list(p1)
            if rng.random() < mut:
                u, v = rng.integers(0, n_nodes, size=2)
                child[u], child[v] = child[v], child[u]
            co = list(fn(np.array([child]))[0])
            offs.append(co)
            z = [min(z[m], co[m]) for m in range(2)]
            n_upd = 0
            for j in rng.permutation(np.array(pool)):
                if n_upd >= nr:
                    break
                if tch(co, w[int(j)]) <= tch(obj[int(j)], w[int(j)]):
                    pop[int(j)], obj[int(j)] = child, co
                    n_upd += 1
        arch = _front(list(arch) + [tuple(o) for o in offs])
        hist.append(arch)
    return pop, obj, hist


def test_full_run_matches_loop_reference_with_identical_random_stream():
    rng = np.random.default_rng(77)
    for k in range(12):
        n_nodes, n_pop = int(rng.integers(5, 9)), int(rng.choice([6, 10, 20]))
        gens, t = int(rng.integers(1, 5)), int(rng.choice([2, 3, 6, 40]))
        cx, mut, delta, nr = float(rng.choice([0.5, 1.0])), float(rng.choice([0.2, 1.0])), float(rng.choice([0.0, 0.9])), int(rng.choice([1, 4, 40]))
        D = A.dist_matrix(rng.random((n_nodes, 2)) * 100)
        fm = rng.uniform(0.6, 3.4, size=(n_nodes, n_nodes))
        fm = (fm + fm.T) / 2

        def fn(pop, D=D, fm=fm):
            return np.array([[sum(D[p[i], p[(i + 1) % len(p)]] for i in range(len(p))),
                              sum(D[p[i], p[(i + 1) % len(p)]] * fm[p[i], p[(i + 1) % len(p)]] for i in range(len(p)))] for p in pop])

        neigh = A.neighborhoods(A.generate_weight_vectors(n_pop), t)
        res = A.run_moead(fn, n_nodes, n_pop, gens, cx, mut, t, delta, nr, k, keep_history=True)
        pop, obj, hist = _ref_run(fn, n_nodes, n_pop, gens, cx, mut, delta, nr, k, neigh)
        assert res.final_population.tolist() == pop
        assert np.allclose(res.final_objectives, obj)
        for h, rh in zip(res.archive_history, hist):
            assert np.allclose(sorted(map(tuple, h.tolist())), rh)


def test_archive_does_not_keep_float_noise_duplicates():
    """Regression: dieselbe Rundtour in anderer Richtung/Drehung liefert Zielwerte, die sich nur um Fließkomma-Rauschen
    unterscheiden; als exakt verschiedene Zeilen blieben beide nicht-dominiert und das Archiv zählte einen Scheinpunkt mehr."""
    base = np.array([[100.0, 200.0], [150.0, 120.0]])
    noisy = np.array([[100.0 + 1e-13, 200.0 - 1e-13]])           # derselbe Punkt, Rauschen in beide Richtungen
    arch = A.update_archive(base, noisy)
    assert len(arch) == 2
    # echte Instanz: eine Tour und ihre Umkehrung dürfen das Archiv nicht verdoppeln
    s = E.Settings(n=8, seed=19)
    fn, n_nodes = E.objective_fn(s)
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(400):
        t = rng.permutation(n_nodes)
        pair = fn(np.stack([t, np.concatenate([t[:1], t[1:][::-1]])]))     # gespiegelt, Knoten 0 bleibt vorn
        if not np.array_equal(pair[0], pair[1]):                 # Rauschen tatsächlich aufgetreten
            hits += 1
            assert np.allclose(pair[0], pair[1], atol=1e-9)
            assert len(A.update_archive(np.empty((0, 2)), pair)) == 1
    assert hits > 0
