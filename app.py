"""MOEA/D - Zerlegung statt Dominanz-Sortierung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, ein KONTRAST zu NSGA-II (kein Fix): MOEA/D
(Zhang & Li, 2007) sortiert nicht nach Pareto-Dominanz, sondern zerlegt das Mehrzielproblem in viele skalare
Unterprobleme (ein Gewichtsvektor je Unterproblem, Tchebycheff-Skalarisierung) und lässt jedes Unterproblem nur mit
seinen Nachbarn (ähnlicher Gewichtsvektor) paaren und konkurrieren. Vehikel ist dieselbe Lieferroute wie
genetic-algorithm-demo/nsga2-demo (Distanz, CO2) - direkt vergleichbar mit deren gemessenen Befunden auf derselben
kleinen Instanz.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import moead_constants as C
from moead_evaluation import Settings, analyse, comparison_experiment, neighborhood_experiment, sweep
from moead_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from moead_visualization import build_archive_size_curve, build_comparison, build_neighborhood_experiment, build_population_scatter, build_sweep

st.set_page_config(page_title="MOEA/D – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _comparison():
    return comparison_experiment()


@st.cache_data(show_spinner=False)
def _neighborhood():
    return neighborhood_experiment()


st.title("🧬 MOEA/D – Zerlegung statt Dominanz-Sortierung")
st.markdown(
    """
NSGA-II und NSGA-III sortieren die Population nach **Pareto-Dominanz**. **MOEA/D** (Zhang & Li, 2007) geht einen ganz
anderen Weg: es **zerlegt** das Mehrzielproblem in viele **skalare Unterprobleme** - eines je Gewichtsvektor, verbunden
über eine **Tchebycheff-Skalarisierung**. Jedes Unterproblem hat eine eigene **Nachbarschaft** (die Unterprobleme mit dem
ähnlichsten Gewichtsvektor) und paart und konkurriert nur dort, nicht mit der ganzen Population. Ein **externes Archiv**
sammelt die nicht-dominierten Lösungen über den ganzen Lauf - die Arbeitspopulation selbst kann eine gute Lösung wieder
verlieren, wenn ihr Unterproblem lokal überschrieben wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "viertes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, ein **Kontrast** zu "
    "[nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/) statt eines Fixes - **ein** Verfahren an einem "
    "wachsenden Beispiel. Vehikel ist dieselbe Lieferroute mit zwei Zielen (Distanz, CO2) wie "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/)/nsga2-demo."
)

with st.expander("So funktioniert MOEA/D", expanded=True):
    st.markdown(
        """
1. **Gewichtsvektoren.** `N` gleichmäßig verteilte Gewichte (Distanz, CO2) - eines je Individuum der Population, jedes
   definiert ein eigenes Unterproblem.
2. **Tchebycheff-Skalarisierung.** Ein Unterproblem bewertet eine Lösung über $\\max_k w_k \\cdot |f_k(x) - z_k|$, mit dem
   Idealpunkt $z$ (bisher bester Wert je Ziel). Verschiedene Gewichte bevorzugen verschiedene Kompromisse.
3. **Nachbarschaft.** Für jeden Gewichtsvektor die `T` ähnlichsten (euklidischer Abstand im Gewichtsraum) - Paarung und
   Ersetzung finden nur dort statt, nicht über die ganze Population.
4. **Crossover + Mutation.** Wie beim GA/NSGA-II (Order Crossover, Tausch-Mutation), Eltern meist aus der Nachbarschaft
   (mit Wahrscheinlichkeit δ), gelegentlich aus der ganzen Population.
5. **Ersetzung.** Ein Nachkomme ersetzt einen Nachbarn, wenn er dessen Unterproblem besser löst (niedrigerer
   Tchebycheff-Wert) - begrenzt auf `nr` Ersetzungen je Nachkomme.
6. **Externes Archiv.** Sammelt die nicht-dominierten Zielwerte über den ganzen Lauf - unabhängig von der
   Arbeitspopulation, die einzelne gute Lösungen wieder verlieren kann.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
    st.markdown("**MOEA/D**")
    pop_size = st.slider("Populationsgröße", *bounds("pop_slider"), key="pop_slider", step=C.POP_STEP, help="= Zahl der Gewichtsvektoren/Unterprobleme.")
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    cx_prob = st.slider("Crossover-Rate", *bounds("cx_slider"), key="cx_slider", step=C.CX_STEP, format="%.2f")
    mut_prob = st.slider("Mutationsrate", *bounds("mut_slider"), key="mut_slider", step=C.MUT_STEP, format="%.2f")
    t = st.slider("Nachbarschaftsgröße T", *bounds("t_slider"), key="t_slider", step=C.T_STEP, help="Wie viele Unterprobleme (nach Gewichtsvektor-Ähnlichkeit) sich ein Unterproblem für Paarung/Ersetzung teilt.")
    neighbor_prob = st.slider("Nachbarschafts-Wahrscheinlichkeit δ", *bounds("neighbor_prob_slider"), key="neighbor_prob_slider", step=C.NEIGHBOR_PROB_STEP, format="%.2f", help="Wie oft Eltern aus der Nachbarschaft statt der ganzen Population gewählt werden.")
    update_limit = st.slider("Ersetzungslimit nr", *bounds("update_limit_slider"), key="update_limit_slider", step=C.UPDATE_LIMIT_STEP, help="Wie viele Nachbarn ein einzelner Nachkomme höchstens ersetzen darf.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des MOEA/D-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "n_slider": int(n_stops), "pop_slider": int(pop_size), "gens_slider": int(generations), "cx_slider": float(cx_prob),
    "mut_slider": float(mut_prob), "t_slider": int(t), "neighbor_prob_slider": float(neighbor_prob), "update_limit_slider": int(update_limit),
    "seed_input": int(seed), "run_seed_input": int(run_seed),
})

settings = Settings(int(n_stops), 0, int(seed), int(pop_size), int(generations), float(cx_prob), float(mut_prob), int(t), float(neighbor_prob), int(update_limit), int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations) - 1
data_key = settings

# --- MOEA/D in Aktion --------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 MOEA/D in Aktion")
if "moead_gen" not in st.session_state or st.session_state.get("moead_gen_owner") != data_key:
    st.session_state["moead_gen"] = n_gens_run
    st.session_state["moead_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 0, n_gens_run, key="moead_gen", help="0 = Startpopulation.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_gens_run == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_gens_run, min(n_gens_run + 1, 40))})


def _render(g):
    gd = result.generations[g]
    archive = result.archive_history[g]
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Generation {g} von {n_gens_run} – Archiv: {len(archive)} Lösungen**")
        c1.plotly_chart(build_population_scatter(gd.objectives, archive), width="stretch", key=f"g_scatter_{g}")
        c2.markdown("**Größe des Archivs**")
        c2.plotly_chart(build_archive_size_curve(result.archive_history[:g + 1]), width="stretch", key=f"g_curve_{g}")


if auto_play:
    for f in _frames():
        _render(f)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was MOEA/D gefunden hat")
m1, m2, m3 = st.columns(3)
m1.metric("Größe des externen Archivs", f"{len(result.archive)}", delta=f"Start {len(result.archive_history[0])}", delta_color="off")
m2.metric("Größe der Arbeitspopulation", f"{len(result.final_population)}")
m3.metric("Generationen × Population", f"{settings.gens} × {settings.pop}")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt die Archivgröße von Population und Nachbarschaftsgröße ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
base_sweep = Settings(n=settings.n, cx=settings.cx, mut=settings.mut, t=settings.t, neighbor_prob=settings.neighbor_prob, update_limit=settings.update_limit, pop=settings.pop)
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: schließt die Linie ab ----------------------------------------------------------------------------------------------

st.subheader("🔬 Wie schlägt sich MOEA/D gegen gewichtete Summe und NSGA-II?")
st.caption(f"Dieselbe kleine Instanz wie genetic-algorithm-demo/nsga2-demo ({C.COMPARISON_N} Stopps, Vehikel-Seed {C.COMPARISON_VEHICLE_SEED}) - dort trafen gewichtete Summe {C.GA_WEIGHTED_SUM_REACHED} von {C.GA_WEIGHTED_SUM_FRONT_SIZE} und NSGA-II {C.NSGA2_REACHED} von {C.GA_WEIGHTED_SUM_FRONT_SIZE} Frontpunkten.")
if st.button("Brute-Force-Front gegen MOEA/D rechnen (dauert etwa 20 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    with st.spinner("Rechne die Brute-Force-Front und mehrere MOEA/D-Läufe..."):
        report = _comparison()
    st.plotly_chart(build_comparison(report), width="stretch", key="comparison_chart")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gewichtete Summe", f"{report['ga_weighted_sum_reached']} von {report['front_size']}")
    c2.metric("NSGA-II", f"{report['nsga2_reached']} von {report['front_size']}")
    c3.metric("MOEA/D, Median", f"{report['reached_median']:.0f} von {report['front_size']}")

st.markdown("---")

# --- Experiment 2: Nachbarschaftsgröße T -----------------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hängt das Archiv von der Nachbarschaftsgröße T ab?")
st.caption("Zu klein: Unterprobleme sind fast isoliert, wenig Zusammenarbeit. Zu groß: die Zerlegung verwässert sich Richtung einer global gemischten Population.")
if st.button("Nachbarschaftsgrößen 3 bis 40 vergleichen (dauert etwa 20 Sekunden)", key="neighborhood_start"):
    st.session_state["neighborhood_on"] = True
if st.session_state.get("neighborhood_on"):
    with st.spinner("Rechne 5 Nachbarschaftsgrößen × 10 Läufe..."):
        rows_n = _neighborhood()
    st.plotly_chart(build_neighborhood_experiment(rows_n), width="stretch", key="neighborhood_chart")
    st.warning(
        "**Ehrlicher Befund:** Über 10 Läufe je T-Wert (Standardvehikel, 60 Gewichtsvektoren, 100 Generationen) ergibt sich "
        "keine saubere U-Form. Die Median-Archivgröße bewegt sich zwischen 4 und 6{,}5 und folgt T nicht monoton "
        "(T=3: 5{,}5 · T=6: 5{,}0 · T=10: 6{,}5 · T=20: 4{,}0 · T=40: 4{,}0) - erkennbar ist nur, dass sehr große T-Werte "
        "eher zu kleineren Archiven führen, nicht aber ein sauberer Mittelbereich-Vorteil. Bei nur 10 Läufen je Wert bleibt "
        "Rauschen ein plausibler Mitgrund; ein einzelner Lauf mit dem Standard-Seed zeigt dagegen den erwarteten Kontrast "
        "deutlich (T=3: Archiv 8, T=40: Archiv 3 - siehe Presets 'Kleine Nachbarschaft'/'Große Nachbarschaft')."
    )

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gleichmäßig verteilte Gewichtsvektoren erzeugen eine gleichmäßige Front** | Bei einer stark nicht-konvexen oder ungleichmäßig gekrümmten Pareto-Front verteilen sich die Lösungen trotzdem ungleichmäßig - ein bekanntes MOEA/D-Problem. | Andere Skalarisierungen (z. B. PBI), adaptive Gewichtsvektoren |
| **Die Arbeitspopulation behält gute Lösungen** | Eine lokal überschriebene Lösung geht ohne externes Archiv verloren, auch wenn sie insgesamt gut war. | Das externe Archiv selbst - ohne es wäre MOEA/D anfälliger dafür |
| **Nachbarschaftsgröße T ist gut gewählt** | Sehr großes T lässt die Ersetzung fast global werden und die Population kollabiert auf wenige Touren (gemessen). Ein sauberer Vorteil eines mittleren T ließ sich über mehrere Seeds hier aber nicht zeigen - siehe Befund oben. | Muss von Hand eingestellt werden, wie bei jedem Regler dieser Linie |
| **Pareto-Dominanz vs. Zerlegung ist eine echte Wahl, kein Fix** | MOEA/D ist als Kontrast zu NSGA-II gedacht, nicht als Verbesserung - beide lösen dieselbe Aufgabe über unterschiedliche Mechanismen. | Kein Nachfolger in dieser Linie geplant |
"""
)
st.caption(
    "MOEA/D schließt die NSGA-II-Familie dieser Linie ab (kein Nachfolger geplant). Vorgänger: "
    "[nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/) und "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/), deren Befunde hier direkt verglichen werden."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zerlegung.** Das Mehrzielproblem $\min (f_1(x), f_2(x))$ wird in $N$ skalare Unterprobleme zerlegt, eines je
Gewichtsvektor $w^{(i)} = (w^{(i)}_1, w^{(i)}_2)$ mit $w^{(i)}_1 + w^{(i)}_2 = 1$, gleichmäßig verteilt:
$w^{(i)}_1 = i / (N-1)$.

**Tchebycheff-Skalarisierung.** $g^{te}(x \mid w, z) = \max_k w_k \cdot |f_k(x) - z_k|$, mit dem Idealpunkt
$z_k = \min_x f_k(x)$ über alle bisher ausgewerteten Lösungen.

**Nachbarschaft.** $B(i) = $ die $T$ Indizes $j$ mit dem kleinsten euklidischen Abstand $\|w^{(i)} - w^{(j)}\|$
(inklusive $i$ selbst).

**Ersetzung.** Ein Nachkomme $y$ (aus Eltern in $B(i)$, mit Wahrscheinlichkeit $\delta$, sonst aus der ganzen Population)
ersetzt bis zu $nr$ Individuen $x_j$, $j \in B(i)$, wenn $g^{te}(y \mid w^{(j)}, z) \le g^{te}(x_j \mid w^{(j)}, z)$.

**Externes Archiv.** Die Vereinigung aller je ausgewerteten Nachkommen, gefiltert auf die nicht-dominierte Teilmenge
(Pareto-Dominanz wie bei NSGA-II/III, aber nur zur Archivpflege, nicht zur Selektion).

Implementiert in `moead_algorithm.py` (Gewichtsvektoren, Nachbarschaften, Tchebycheff, Hauptschleife, Archiv),
`moead_scenario.py` (Vehikel), `moead_evaluation.py` (Kennzahlen, Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
