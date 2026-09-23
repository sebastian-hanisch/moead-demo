# 🧬 MOEA/D – Zerlegung statt Dominanz-Sortierung

Viertes Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Kontrast zu [nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/)
und [nsga3-demo](https://sebastianhanisch-nsga3-demo.streamlit.app/): MOEA/D (Zhang & Li, 2007) sortiert nicht nach
Pareto-Dominanz, sondern **zerlegt** das Mehrziel-Problem in viele skalare Unterprobleme (je ein Gewichtsvektor), die sich
**nachbarschaftsbasiert** gegenseitig Lösungen weitergeben. Dieselbe Lieferroute wie die beiden Vorgänger, wieder mit
zwei Zielen (Distanz, CO2) - MOEA/D ist hier als echt anderer Mechanismus gedacht, nicht als Viele-Ziele-Fix wie NSGA-III.

## Warum dieses Problem

NSGA-II und NSGA-III sortieren die Population nach Pareto-Dominanz und ordnen ihr Diversität über Crowding-Distance bzw.
Referenzpunkte zu - beide arbeiten mit *einer* Population, die als Ganzes gegen das Mehrziel-Problem antritt. MOEA/D geht
anders vor: Es zerlegt das Problem vorab in *N* skalare Unterprobleme (Tchebycheff-Skalarisierung mit je einem
Gewichtsvektor), verteilt sie auf ein Nachbarschaftsnetz (ähnliche Gewichtsvektoren = Nachbarn) und lässt jedes
Unterproblem im Wesentlichen seinen eigenen, lokalen genetischen Algorithmus fahren - Eltern und Ersetzungskandidaten
kommen bevorzugt aus der Nachbarschaft. Ein externes Archiv sammelt die bisher gefundenen nicht-dominierten Lösungen,
weil die einzelnen Unterprobleme selbst keine globale Pareto-Front garantieren.

## Modell

Dieselbe Lieferroute wie genetic-algorithm-demo/nsga2-demo: ein Depot in der Mitte und *n* Kundenstopps in einem
100 × 100-km-Gebiet, euklidische Entfernungen, CO2-Faktor je Straßenabschnitt (0,6–3,4). **`moead_scenario.generate_perm`
reproduziert die Vehikel-Erzeugung wortgleich** - die kleine Vergleichsinstanz (8 Stopps, Vehikel-Seed 19) ist bitidentisch
zu den beiden Vorgängern, direkt zitierbar.

## Methodik

MOEA/D-Kern (`moead_algorithm.py`, kein NSGA-Kern kopiert - andere Mechanik): gleichmäßig verteilte Gewichtsvektoren
(`generate_weight_vectors`), Nachbarschaften über euklidischen Abstand im Gewichtsraum (`neighborhoods`), Tchebycheff-
Skalarisierung (`tchebycheff`, exakt gegen `pymoo.decomposition.tchebicheff.Tchebicheff` geprüft). Die Hauptschleife
(`run_moead`) wählt pro Unterproblem mit Wahrscheinlichkeit δ Eltern aus der Nachbarschaft (sonst aus der ganzen
Population), erzeugt einen Nachkommen (Order Crossover + Tausch-Mutation, wortgleich aus nsga2-demo kopiert), aktualisiert
den Idealpunkt und ersetzt Nachbarn, deren Tchebycheff-Wert der Nachkomme unterbietet - begrenzt auf `nr` Ersetzungen je
Nachkomme (wie im Originalpapier, gegen zu schnelle Übernahme durch einen einzelnen guten Nachkommen). Ein externes
Archiv (`update_archive`, nicht-dominierte Menge über alle bisher erzeugten Nachkommen) entspricht "Front 1" bei
NSGA-II/III.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Schließt sich die NSGA-II-Familie dieser Linie ab? | Auf der identischen 8-Stopp-Instanz: gewichtete Summe 4 von 8, NSGA-II 6 von 8, MOEA/D im Median **6 von 8** (über 20 Läufe) - MOEA/D erreicht hier ungefähr NSGA-II-Niveau, klar vor der gewichteten Summe, über einen komplett anderen Mechanismus | `test_comparison_experiment_headline_claims` |
| Bewirkt eine kleine Nachbarschaftsgröße T mehr Vielfalt, eine große mehr Konvergenz? | Einzelner Standard-Lauf: T=3 → Archiv 8, Population noch auf 7 verschiedene Touren verteilt. T=40 → Archiv 3, Population auf 3 Touren kollabiert - der erwartete Kontrast zeigt sich deutlich | `test_kleine_nachbarschaft_preset_claims`, `test_grosse_nachbarschaft_preset_claims` |
| Gilt dieser Kontrast auch über mehrere Seeds gemittelt? | **Nein, nicht sauber.** Median-Archivgröße über 10 Läufe je T-Wert: T=3: 5,5 · T=6: 5,0 · T=10: 6,5 · T=20: 4,0 · T=40: 4,0 - kein monotoner Zusammenhang, nur dass sehr große T eher zu kleineren Archiven neigt | `test_neighborhood_experiment_structural_claims` |
| Stimmt die eigene Tchebycheff-Skalarisierung mit der Literatur überein? | Exakte Übereinstimmung mit `pymoo` über 15 Zufallsinstanzen | `test_tchebycheff_matches_pymoo` |

## Ehrliche Grenzen

- **Der Nachbarschaftsgrößen-Befund ist gemessen widersprüchlich**: Ein einzelner Lauf mit dem Standard-Seed zeigt den aus
  der Theorie erwarteten Kontrast (klein-T = vielfältig, groß-T = konvergiert) sehr deutlich; der Median über zehn Seeds
  zeigt dagegen keine saubere U-Form. Beides wird hier nebeneinander berichtet, nicht das bequemere Ergebnis ausgewählt.
- **Gleichmäßig verteilte Gewichtsvektoren erzeugen keine automatisch gleichmäßige Front** - bei einer stark
  nicht-konvexen oder ungleichmäßig gekrümmten Pareto-Front ein bekanntes MOEA/D-Problem (hier nicht gezielt geprüft).
- **Ohne externes Archiv gingen gute Lösungen verloren** - die Arbeitspopulation selbst überschreibt lokal, auch wenn eine
  überschriebene Lösung insgesamt gut war.
- **Kein Nachfolger in dieser Linie geplant** - MOEA/D ist als Kontrast zu NSGA-II gedacht, nicht als Fix; damit schließt
  dieses Stück die NSGA-II-Familie der Populations-Metaheuristiken-Linie ab.

## Tests

68 Tests (`pytest tests/ -v`): Gewichtsvektoren/Nachbarschaften per Handrechnung, Tchebycheff-Skalarisierung exakt gegen
`pymoo` geprüft (15 Zufallsinstanzen), externes Archiv gegen konstruierte Beispiele, MOEA/D findet auf einer sehr kleinen
Instanz nachweislich die Mehrheit der Brute-Force-Front, Szenario-Erzeugung bitidentisch zu nsga2-demo geprüft,
AppTest-Rauchtests (jedes Preset, Generation-Slider inkl. Abspielen, Permalink-Grenzen, beide Experimente + Sweep auf
Abruf) und `test_claims.py` (jede Zahl aus diesem README, mit CI-robusten Bändern für Einzellauf-Kennzahlen - siehe
`feedback_ci_platform_robust_tests.md`).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `moead_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `moead_presets.py` | Permalink/Presets-Mechanik |
| `moead_scenario.py` | Vehikel-Erzeuger (Lieferroute, Distanz/CO2), wortgleich zu nsga2-demo |
| `moead_algorithm.py` | MOEA/D-Kern (Gewichtsvektoren, Nachbarschaften, Tchebycheff, Hauptschleife, Archiv) |
| `moead_evaluation.py` | Kennzahlen, Brute-Force-Referenz, Vergleichs- und Nachbarschaftsgrößen-Experiment, Sweep |
| `moead_visualization.py` | Plotly-Abbildungen (Streudiagramm, Vergleiche, Boxplots) |

## Bewusst nicht umgesetzt

- Andere Skalarisierungen (z. B. PBI) oder adaptive Gewichtsvektoren - jenseits des Originalpapiers.
- Mehr als zwei Ziele - MOEA/Ds Kontrast bezieht sich hier bewusst auf denselben Zielumfang wie NSGA-II, nicht auf viele
  Ziele wie NSGA-III.
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
