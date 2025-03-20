# README: Analyse von Verpackungsmaschinen-Daten

## 📚 Projektbeschreibung
Dieses Projekt (Prototyp) analysiert Daten von Verpackungsmaschinen, um die Qualität von Verpackungen vorherzusagen und zu optimieren. Es kombiniert **klassische Machine-Learning-Modelle** mit **kausaler Inferenz**, um Ursachen für Qualitätsmängel zu identifizieren und Optimierungsstrategien vorzuschlagen.

---

## 🔄 Workflow & Technologien
### ✅ **Erstellung von Maschinendaten (`create_machine_data.ipynb`)**
- **Ziel**: Erstellung und Analyse von synthetischen Produktionsdaten für Verpackungsmaschinen.
- **Technologien**: `pandas`, `numpy`, `matplotlib`
- **Schritte**:
    1. **Daten generieren**: Simulation von Maschinenparametern und Verpackungsqualitäten.
    2. **Datenaufbereitung**: Bereinigung und Normalisierung der Daten.
    3. **Explorative Datenanalyse (EDA)**: Untersuchung von Korrelationen zwischen Maschinenparametern und Qualität.

---

### 🔬 **Kausalanalyse & Optimierung (`causal_ai_with_machine_data.ipynb`)**
- **Ziel**: Nutzung kausaler Modelle zur Identifikation von Einflüssen auf die Verpackungsqualität und Vorschlag von Optimierungsmaßnahmen.
- **Technologien**: `DoWhy`, `EconML`, `networkx`, `matplotlib`, `pandas`, `scikit-learn`, `joblib`, `plotly`, `streamlit`
- **Schritte**:
    1. **Datenvorbereitung**
        - Laden der Produktionsdaten und Normalisierung mittels `StandardScaler`.
        - Definition der relevanten Behandlungsvariablen (`treatments`) und Zielvariable (`Qualität_Numerisch`).
    2. **Kausale Modellierung mit `DoWhy`**
        - Aufbau eines kausalen Graphen mit `networkx`, um Zusammenhänge zwischen Maschinenparametern zu visualisieren.
        - Identifikation von **direkten Einflussfaktoren und Confoundern**.
        - **DoWhy-Ansatz:**
            - Definition eines kausalen Modells mit `CausalModel`.
            - Identifikation des kausalen Effekts mittels **Backdoor-Kriterium**.
            - Schätzung des kausalen Effekts mit `backdoor.linear_regression`.
            - Durchführung von **Refutation Tests**, um die Robustheit der Analyse zu überprüfen.
    3. **Erweiterte Kausalmodelle mit `EconML`**
        - Nutzung von `LinearDML` zur kausalen Schätzung unter Berücksichtigung von Confoundern.
        - Anwendung von `CausalForestDML` zur Modellierung **heterogener Behandlungseffekte**, um personalisierte Optimierungsvorschläge abzuleiten.
        - Vergleich verschiedener Kausalmodelle zur Bestimmung der besten Methode für die Verpackungsoptimierung.
    4. **Optimierung & Handlungsempfehlungen**
        - Berechnung optimaler Maschinenparameter zur Qualitätsverbesserung mit `CausalForestDML`.
        - Identifikation der **wichtigsten Parameter** mit dem größten Einfluss auf die Qualität.
        - Durchführung von **Counterfactual-Analysen**, um alternative Szenarien für die Optimierung der Verpackungsprozesse zu simulieren.
        - Visualisierung der kausalen Effekte mit `plotly`.
    5. **Modellentwicklung & Deployment**
        - Training eines `RandomForestRegressor` zur Qualitätsvorhersage.
        - Vergleich der kausalen Modellierung mit nicht-kausalen ML-Methoden zur Beurteilung der **zusätzlichen Erkenntnisse aus der Kausalanalyse**.
        - Speicherung der Modelle (`joblib`) für spätere Anwendungen im Dashboard.
    6. **Entwicklung eines interaktiven Dashboards (`dashboard.py`)**
        - Implementierung einer **Streamlit**-Anwendung zur Qualitätsvorhersage und Ursachenanalyse.
        - **Funktionen des Dashboards:**
            - **Qualitätsvorhersage:** Nutzung des `RandomForestRegressor`, um die Verpackungsqualität auf Basis der aktuellen Maschinenparameter vorherzusagen.
            - **Parameteroptimierung:** Durchführung einer **Gegenfaktischen Analyse (`Counterfactual Analysis`)** mit `CausalForestDML`, um zu zeigen, wie sich Änderungen einzelner Parameter auf die Qualität auswirken.
            - **Ursachenanalyse:** Identifikation der Hauptfaktoren, die Qualitätsprobleme verursachen, durch eine Kombination aus `DoWhy`-Modellen und `EconML`-Schätzungen.
            - **Datenexploration:** Visualisierung von Korrelationen zwischen Parametern und Qualität mithilfe von `plotly`.

---

## 🔍 Erkenntnisse
✅ **Maschinenparameter beeinflussen die Verpackungsqualität signifikant** (z. B. Temperatur, Druck, Geschwindigkeit).  
✅ **Kausale Analysen zeigen den direkten Einfluss einzelner Parameter** und ermöglichen gezielte Optimierungen.  
✅ **Vergleich mit klassischen ML-Methoden zeigt, dass kausale Analysen bessere Erklärbarkeit und optimierte Handlungsempfehlungen liefern.**  
✅ **Das entwickelte Dashboard erlaubt eine interaktive Qualitätsoptimierung und Ursachenanalyse in Echtzeit.**  
✅ **Datengetriebene Entscheidungen verbessern Produktionsprozesse nachhaltig**.

---

## 📊 Nächste Schritte
1. **Erweiterung des Dashboards** um weitere Visualisierungen und Optimierungsalgorithmen.
2. **Integration zusätzlicher Datenquellen** zur Validierung der kausalen Effekte.
3. **Test und Deployment des Modells** in einer realen Produktionsumgebung zur Überprüfung der Optimierungsvorschläge.
4. **Erweiterung der kausalen Modelle** durch hybride Methoden zur Kombination von `DoWhy` und `EconML` für präzisere Empfehlungen.

