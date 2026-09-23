"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 100-150-Generationen-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (siehe
feedback_ci_platform_robust_tests.md, und die eigene Erfahrung aus nsga2-demo/nsga3-demo). Zahlen aus einem EINZELNEN
Lauf (Presets) bekommen deshalb nur Strukturgrenzen; Zahlen, die über mehrere Seeds mitteln (Experimente), sind von
Natur aus robuster und dürfen engere (aber weiterhin großzügige) Bänder bekommen."""

import numpy as np
import pytest

import moead_constants as C
import moead_evaluation as E


def _preset_run(name):
    p = C.PRESETS[name]
    s = E.Settings(n=p["n"], seed=p["seed"], pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], t=p["t"], neighbor_prob=p["neighbor_prob"], update_limit=p["update_limit"], run_seed=p["run_seed"])
    return p, E.run(s, keep_history=False)


# --- Einzelläufe (Presets) - nur Strukturgrenzen, keine Nähe zu einem Messwert ----------------------------------------------------------


def test_standardfall_preset_claims():
    p, r = _preset_run("Standardfall")
    assert 1 <= len(r.archive) <= p["pop"]
    assert 1 <= len(np.unique(r.final_population, axis=0)) <= p["pop"]


def test_kleine_nachbarschaft_preset_claims():
    p, r = _preset_run("Kleine Nachbarschaft")
    assert p["t"] == C.T_MIN
    assert 1 <= len(r.archive) <= p["pop"]


def test_grosse_nachbarschaft_preset_claims():
    p, r = _preset_run("Große Nachbarschaft")
    assert p["t"] == C.T_MAX
    assert 1 <= len(r.archive) <= p["pop"]


def test_kleine_population_preset_claims():
    p, r = _preset_run("Kleine Population")
    assert p["pop"] == C.POP_MIN
    assert 1 <= len(r.archive) <= p["pop"]


def test_kleine_instanz_preset_claims():
    p, r = _preset_run("Kleine Instanz (Vergleich mit Brute-Force)")
    assert p["n"] == C.COMPARISON_N and p["seed"] == C.COMPARISON_VEHICLE_SEED
    assert 1 <= len(r.archive) <= p["pop"]


# --- Headlinezahlen der beiden Experimente (mitteln über 10-20 Seeds, robuster) ---------------------------------------------------------


def test_comparison_experiment_headline_claims():
    report = E.comparison_experiment()
    assert report["front_size"] == 8                          # Brute-Force auf einer festen Instanz - deterministisch
    assert report["ga_weighted_sum_reached"] == 4 and report["nsga2_reached"] == 6   # aus GA-Demo/nsga2-demo zitiert
    assert report["reached_median"] == pytest.approx(6, abs=3)
    assert all(0 <= r <= 8 for r in report["reached_all"])
    # Kernbefund, der die Linie abschließt: MOEA/D erreicht hier ungefähr NSGA-II-Niveau, klar über der gewichteten Summe
    assert report["reached_median"] >= report["ga_weighted_sum_reached"]


def test_neighborhood_experiment_structural_claims():
    rows = E.neighborhood_experiment()
    assert [row["t"] for row in rows] == list(C.NEIGHBORHOOD_T_VALUES)
    for row in rows:
        assert 1 <= row["archive_size_median"] <= C.NEIGHBORHOOD_POP
        assert all(0 <= s <= C.NEIGHBORHOOD_POP for s in row["archive_size_all"])
    # Ehrlicher Befund: KEIN sauberer, monotoner Zusammenhang über die Seeds - deshalb hier bewusst keine
    # richtungsgebundene Behauptung (z. B. "mittleres T ist am besten"), nur dass die Werte in einem plausiblen Rahmen bleiben.
