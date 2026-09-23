"""Konstanten der MOEA/D-Demo: Vehikel (wie nsga2-demo, 2 Ziele), MOEA/D-Regler, Presets (Presets folgen nach den Messungen)."""

# --- Vehikel: Lieferroute (wie nsga2-demo/genetic-algorithm-demo) --------------------------------------------------------------------------

AREA = 100.0
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0
CLUSTER_MARGIN = 12.0
N_MIN, N_MAX, DEFAULT_N, N_STEP = 8, 100, 30, 2

# CO2-Faktor je Straßenabschnitt (Kante), unabhängig von der Distanz - Bereich bewusst identisch zu nsga2-demo, damit die
# kleine Vergleichsinstanz (n=8, Seed 19) bitidentisch bleibt.
CO2_FACTOR_LO, CO2_FACTOR_HI = 0.6, 3.4

N_OBJ = 2                          # bewusst wie NSGA-II (Distanz, CO2) - MOEA/D ist hier ein Kontrast, kein Viele-Ziele-Fix
OBJECTIVE_LABELS = ("Distanz", "CO2")

# --- MOEA/D ------------------------------------------------------------------------------------------------------------------------------

POP_MIN, POP_MAX, DEFAULT_POP, POP_STEP = 20, 200, 60, 10      # = Zahl der Gewichtsvektoren/Unterprobleme
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 400, 150, 10
CX_MIN, CX_MAX, DEFAULT_CX, CX_STEP = 0.0, 1.0, 0.9, 0.05
MUT_MIN, MUT_MAX, DEFAULT_MUT, MUT_STEP = 0.0, 1.0, 0.2, 0.05
T_MIN, T_MAX, DEFAULT_T, T_STEP = 3, 40, 10, 1                  # Nachbarschaftsgröße
NEIGHBOR_PROB_MIN, NEIGHBOR_PROB_MAX, DEFAULT_NEIGHBOR_PROB, NEIGHBOR_PROB_STEP = 0.0, 1.0, 0.9, 0.05   # δ: Elternwahl aus der Nachbarschaft statt der ganzen Population
UPDATE_LIMIT_MIN, UPDATE_LIMIT_MAX, DEFAULT_UPDATE_LIMIT, UPDATE_LIMIT_STEP = 1, 40, 4, 1                # nr: max. Ersetzungen je Nachkomme
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_RUN_SEED = 7

# --- Kleine Vergleichsinstanz: identisch zur Pareto-Front-Instanz der genetic-algorithm-demo/nsga2-demo -------------------------------------

COMPARISON_N = 8
COMPARISON_VEHICLE_SEED = 19
GA_WEIGHTED_SUM_FRONT_SIZE = 8      # gemessen in genetic-algorithm-demo
GA_WEIGHTED_SUM_REACHED = 4         # gemessen in genetic-algorithm-demo
NSGA2_REACHED = 6                   # gemessen in nsga2-demo (comparison_experiment)
COMPARISON_SEEDS = tuple(range(1000000, 1000020))
COMPARISON_POP, COMPARISON_GENS = 60, 150

# --- Nachbarschaftsgrößen-Experiment (eigener Regler) ------------------------------------------------------------------------------------

NEIGHBORHOOD_T_VALUES = (3, 6, 10, 20, 40)
NEIGHBORHOOD_SEEDS = tuple(range(1100000, 1100010))
NEIGHBORHOOD_POP, NEIGHBORHOOD_GENS = 60, 100

SWEEP_SEEDS = tuple(range(1200000, 1200005))
SWEEP_VALUES = {"pop": (20, 40, 60, 100, 150), "t": (3, 6, 10, 20, 40)}
SWEEP_LABELS = {"pop": "Populationsgröße (Gewichtsvektoren)", "t": "Nachbarschaftsgröße T"}


def _preset(n=DEFAULT_N, pop=DEFAULT_POP, gens=DEFAULT_GEN, cx=DEFAULT_CX, mut=DEFAULT_MUT, t=DEFAULT_T, neighbor_prob=DEFAULT_NEIGHBOR_PROB, update_limit=DEFAULT_UPDATE_LIMIT, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    return {"n": n, "pop": pop, "gens": gens, "cx": cx, "mut": mut, "t": t, "neighbor_prob": neighbor_prob, "update_limit": update_limit, "seed": seed, "run_seed": run_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Kleine Nachbarschaft": _preset(t=3),
    "Große Nachbarschaft": _preset(t=40),
    "Kleine Population": _preset(pop=20),
    "Kleine Instanz (Vergleich mit Brute-Force)": _preset(n=COMPARISON_N, seed=COMPARISON_VEHICLE_SEED, pop=COMPARISON_POP, gens=COMPARISON_GENS),
}
PRESET_HELP = {
    "Standardfall": "T=10, 60 Gewichtsvektoren: Archiv am Ende 6 nicht-dominierte Touren, aber die Population selbst ist auf nur 3 verschiedene Touren konvergiert (gemessen, Standard-Seed).",
    "Kleine Nachbarschaft": "T=3: Unterprobleme bleiben unabhängiger - Archiv 8, Population noch auf 7 verschiedene Touren verteilt (gemessen, sonst wie Standardfall).",
    "Große Nachbarschaft": "T=40 (fast die ganze Population): Ersetzung wirkt fast global - Archiv 3, Population auf nur 3 verschiedene Touren kollabiert (gemessen, sonst wie Standardfall).",
    "Kleine Population": "Nur 20 Gewichtsvektoren: Front wird grob abgetastet - Archiv 3, Population auf 2 verschiedene Touren kollabiert (gemessen).",
    "Kleine Instanz (Vergleich mit Brute-Force)": "Identische 8-Stopp-Instanz wie im Kopfexperiment: Brute-Force-Front hat 8 Punkte, MOEA/D erreicht im Median 6/8 über 20 Läufe - genau wie NSGA-II, klar vor gewichteter Summe (4/8).",
}
