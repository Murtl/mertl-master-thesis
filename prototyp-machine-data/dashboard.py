# dashboard.py - Streamlit Dashboard für Qualitätsoptimierung
import torch
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

torch.classes.__path__ = [] # add this line to manually set it to empty.

# Laden der gespeicherten Modelle und Daten
@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_prediction_model.pkl')
    cf_model = joblib.load('causal_effect_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    treatments = joblib.load('treatment_variables.pkl')
    features = joblib.load('feature_names.pkl')
    all_features = joblib.load('all_features.pkl')[0]
    
    # Laden der ursprünglichen Daten für Referenzwerte
    data = pd.read_csv('verpackungsmaschine_datensatz_5000.csv')
    
    return rf_model, cf_model, scaler, treatments, features, all_features, data

rf_model, cf_model, scaler, treatments, features, all_features, data = load_models()

# Dashboard-Titel
st.title('CausalAI - Qualitätsoptimierung für Verpackungsmaschinen')
st.markdown('Dieses Dashboard unterstützt bei der Analyse und Optimierung der Verpackungsqualität.')

# Seitenleiste mit Tabs
page = st.sidebar.selectbox(
    "Navigation",
    ["Qualitätsvorhersage", "Parameteroptimierung", "Ursachenanalyse", "Datenexploration"],
    key="navigation"
)

# Hilfsfunktion für Ursachenanalyse (aus Hauptskript)
def analyze_quality_issues(input_data, actual_quality=None):
    # Feature-Extraktion
    X_instance = pd.DataFrame([input_data], columns=features)
    X_instance_scaled = scaler.transform(X_instance)
    current_treatments = np.array([input_data[t] for t in treatments]).reshape(1, -1)
    
    # Vorhersage der aktuellen Qualität falls nicht angegeben
    if actual_quality is None:
        all_features_values = [input_data[f] for f in all_features]
        actual_quality = rf_model.predict(
            pd.DataFrame([all_features_values], columns=all_features)
        )[0]
    
    # Sammeln der Verbesserungsvorschläge
    improvements = []
    expected_quality_gain = 0
    
    for t_idx, treatment in enumerate(treatments):
        # Berechnen des optimalen Werts
        t_min = data[treatment].min()
        t_max = data[treatment].max()
        t_current = current_treatments[0, t_idx]
        
        # Erzeugen von 15 möglichen Werten zwischen Min und Max
        t_values = np.linspace(t_min, t_max, 15)
        
        # Für jeden möglichen Wert den Effect abschätzen
        cf_effects = []
        for t_value in t_values:
            new_treatment = current_treatments.copy()
            new_treatment[0, t_idx] = t_value
            effect = cf_model.effect(X_instance_scaled, T0=current_treatments, T1=new_treatment)
            cf_effects.append(effect[0])

            # Finden des optimalen Werts
        optimal_idx = np.argmax(cf_effects)
        optimal_value = t_values[optimal_idx]
        quality_gain = cf_effects[optimal_idx]
        
        # Nur signifikante Verbesserungen berücksichtigen (> 0.05)
        if abs(quality_gain) > 0.05:
            improvements.append({
                'parameter': treatment,
                'current_value': t_current,
                'optimal_value': optimal_value,
                'quality_gain': quality_gain,
                'percent_change': ((optimal_value - t_current) / t_current * 100) if t_current != 0 else float('inf')
            })
            expected_quality_gain += quality_gain
    
    # Sortieren nach dem größten Effekt
    improvements.sort(key=lambda x: abs(x['quality_gain']), reverse=True)
    
    return {
        'current_quality': actual_quality,
        'expected_quality': actual_quality + expected_quality_gain,
        'improvements': improvements
    }

# Funktion zur Generierung eines Counterfactual-Plots
def generate_cf_plot(input_data, treatment_idx):
    # Feature-Extraktion
    X_instance = pd.DataFrame([input_data], columns=features)
    X_instance_scaled = scaler.transform(X_instance)
    current_treatments = np.array([input_data[t] for t in treatments]).reshape(1, -1)
    
    # Aktuelle Vorhersage
    all_features_values = [input_data[f] for f in all_features]
    current_quality = rf_model.predict(
        pd.DataFrame([all_features_values], columns=all_features)
    )[0]
    
    # Parameter und aktuelle Werte
    treatment = treatments[treatment_idx]
    t_min = data[treatment].min()
    t_max = data[treatment].max()
    t_current = current_treatments[0, treatment_idx]
    
    # Erzeugen von 20 möglichen Werten zwischen Min und Max
    t_values = np.linspace(t_min, t_max, 20)
    cf_effects = []
    
    for t_value in t_values:
        new_treatment = current_treatments.copy()
        new_treatment[0, treatment_idx] = t_value
        effect = cf_model.effect(X_instance_scaled, T0=current_treatments, T1=new_treatment)
        #cf_effects.append(effect[treatment_idx])
        cf_effects.append(effect[0] if effect.ndim == 1 else effect[:, treatment_idx])

    # Optimaler Wert finden
    optimal_idx = np.argmax(cf_effects)
    optimal_value = t_values[optimal_idx]
    
    # Plotly-Grafik erstellen
    fig = go.Figure()
    
    # Counterfactual-Linie
    fig.add_trace(go.Scatter(
        x=t_values, 
        y=cf_effects,
        mode='lines+markers',
        name='Vorhergesagte Qualität'
    ))
    
    # Aktuelle Position markieren
    fig.add_vline(x=t_current, line_dash="dash", line_color="red",
                  annotation_text=f"Aktuell: {t_current:.2f}")
    
    # Optimale Position markieren
    fig.add_vline(x=optimal_value, line_dash="dash", line_color="green",
                  annotation_text=f"Optimal: {optimal_value:.2f}")
    
    fig.update_layout(
        title=f'Counterfactual-Analyse: Effekt von {treatment} auf die Qualität',
        xaxis_title=treatment,
        yaxis_title='Vorhergesagte Qualität',
        height=500
    )
    
    return fig

if page == "Qualitätsvorhersage":
    st.header("Qualitätsvorhersage")
    st.write("Geben Sie die Prozessparameter ein, um die zu erwartende Qualität vorherzusagen.")
    
    # Eingabefelder für alle Parameter erstellen
    st.subheader("Prozessparameter")
    
    # Zwei Spalten Layout
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    # Eingabefelder für die Behandlungsvariablen
    with col1:
        st.markdown("**Hauptparameter:**")
        for t in treatments:
            min_val = data[t].min()
            max_val = data[t].max()
            mean_val = data[t].mean()
            
            input_data[t] = st.slider(
                f"{t}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                key=f"opt_{t}"
            )
    
    # Eingabefelder für die übrigen Merkmale
    with col2:
        st.markdown("**Umgebungs- und Produktparameter:**")
        for f in features:
            min_val = data[f].min()
            max_val = data[f].max()
            mean_val = data[f].mean()
            
            input_data[f] = st.slider(
                f"{f}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                key=f"opt_{f}"
            )
    
    # Vorhersagen, wenn der Nutzer auf die Schaltfläche klickt
    if st.button("Qualität vorhersagen"):
        # Alle Merkmale für die Vorhersage sammeln
        all_features_values = [input_data[f] for f in all_features]
        
        # Qualität vorhersagen
        predicted_quality = rf_model.predict(
            pd.DataFrame([all_features_values], columns=all_features)
        )[0]
        
        # Ampelfarbgebung basierend auf der Qualität
        if predicted_quality >= 8.0:
            color = "green"
            quality_text = "Ausgezeichnet"
        elif predicted_quality >= 6.5:
            color = "orange"
            quality_text = "Akzeptabel"
        else:
            color = "red"
            quality_text = "Mangelhaft"
        
        # Anzeigen der Vorhersage mit Farbmarkierung
        st.markdown(f"### Vorhergesagte Qualität: <span style='color:{color}'>{predicted_quality:.2f} ({quality_text})</span>", unsafe_allow_html=True)
        
        # Einfaches Gauge-Diagramm zur Visualisierung
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_quality,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Qualitätsbewertung", 'font': {'size': 24}},
            delta = {'reference': 8.0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 6.5], 'color': 'lightpink'},
                    {'range': [6.5, 8.0], 'color': 'lightyellow'},
                    {'range': [8.0, 10], 'color': 'lightgreen'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6.5}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Anzeigen der empfohlenen Verbesserungen
        st.subheader("Optimierungspotenzial")
        analysis = analyze_quality_issues(input_data, predicted_quality)
        
        if len(analysis['improvements']) > 0:
            improvement_df = pd.DataFrame(analysis['improvements'])
            improvement_df.columns = ['Parameter', 'Aktueller Wert', 'Optimaler Wert', 
                                   'Qualitätsgewinn', '% Änderung']
            
            st.dataframe(improvement_df)
            
            st.markdown(f"**Erwartete Qualität nach Optimierung: {analysis['expected_quality']:.2f}**")
        else:
            st.write("Keine signifikanten Verbesserungsmöglichkeiten gefunden.")

elif page == "Parameteroptimierung":
    st.header("Parameteroptimierung")
    st.write("Untersuchen Sie, wie sich Änderungen an einzelnen Parametern auf die Qualität auswirken.")
    
    # Input-Parameter wie bei der Qualitätsvorhersage
    st.subheader("Aktuelle Prozessparameter")
    
    # Zweispaltig
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.markdown("**Hauptparameter:**")
        for t in treatments:
            min_val = data[t].min()
            max_val = data[t].max()
            mean_val = data[t].mean()
            
            input_data[t] = st.slider(
                f"{t}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                key=f"opt_{t}",
            )
    
    with col2:
        st.markdown("**Umgebungs- und Produktparameter:**")
        for f in features:
            min_val = data[f].min()
            max_val = data[f].max()
            mean_val = data[f].mean()
            
            input_data[f] = st.slider(
                f"{f}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                key=f"opt_{f}",
            )
    
    # Auswahl des zu optimierenden Parameters
    st.subheader("Parameter für die Optimierungsanalyse")
    selected_treatment = st.selectbox(
        "Wählen Sie einen Parameter zur Analyse:",
        treatments,
        key="choose_treatment"
    )
    
    treatment_idx = treatments.index(selected_treatment)
    
    if st.button("Parametereffekt analysieren"):
        # Counterfactual-Plot erstellen
        st.subheader(f"Effekt von {selected_treatment} auf die Qualität")
        fig = generate_cf_plot(input_data, treatment_idx)
        st.plotly_chart(fig, use_container_width=True)
        
        # Berechnen und Anzeigen der Optimalwerte für alle Parameter
        st.subheader("Optimale Parameterwerte")
        
        # Analyse durchführen
        analysis = analyze_quality_issues(input_data)
        
        if len(analysis['improvements']) > 0:
            # Erstellen eines DataFrames für die Darstellung
            improvement_df = pd.DataFrame(analysis['improvements'])
            improvement_df.columns = ['Parameter', 'Aktueller Wert', 'Optimaler Wert', 
                                   'Qualitätsgewinn', '% Änderung']
            
            improvement_df['% Änderung'] = improvement_df['% Änderung'].map('{:.1f}%'.format)
            improvement_df['Qualitätsgewinn'] = improvement_df['Qualitätsgewinn'].map('{:.3f}'.format)
            improvement_df['Aktueller Wert'] = improvement_df['Aktueller Wert'].map('{:.2f}'.format)
            improvement_df['Optimaler Wert'] = improvement_df['Optimaler Wert'].map('{:.2f}'.format)
            
            st.dataframe(improvement_df)
            
            # Visualisierung der Qualitätsgewinne
            if len(improvement_df) > 1:
                fig = px.bar(improvement_df, 
                           x='Parameter', 
                           y='Qualitätsgewinn',
                           title='Erwarteter Qualitätsgewinn durch Parameteroptimierung')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Keine signifikanten Verbesserungsmöglichkeiten gefunden.")

elif page == "Ursachenanalyse":
    st.header("Ursachenanalyse für Qualitätsprobleme")
    st.write("Geben Sie die aktuellen Prozessparameter und die tatsächliche Qualität ein, um eine Ursachenanalyse durchzuführen.")
    
    # Option zum Hochladen von CSV-Daten oder manueller Eingabe
    input_method = st.radio(
        "Datenquelle für die Analyse",
        ["CSV-Datei hochladen", "Manuelle Eingabe"]
    )
    
    if input_method == "CSV-Datei hochladen":
        uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type="csv")
        
        if uploaded_file is not None:
            # Daten einlesen
            upload_df = pd.read_csv(uploaded_file)
            
            # Überprüfen, ob alle benötigten Spalten vorhanden sind
            missing_cols = [col for col in all_features + ['Qualität_Numerisch'] if col not in upload_df.columns]
            
            if missing_cols:
                st.error(f"Folgende erforderliche Spalten fehlen in der CSV-Datei: {', '.join(missing_cols)}")
            else:
                st.success(f"{len(upload_df)} Datensätze erfolgreich geladen.")
                
                # Auswahl des zu analysierenden Datensatzes
                if len(upload_df) > 1:
                    st.subheader("Wählen Sie einen Datensatz für die Analyse")
                    
                    # Anzeigen der Daten mit Qualität-Spalte zuerst für bessere Übersicht
                    display_cols = ['Qualität_Numerisch'] + [c for c in upload_df.columns if c != 'Qualität_Numerisch']
                    st.dataframe(upload_df[display_cols])
                    
                    row_index = st.number_input("Zeilennummer auswählen", 
                                             min_value=0, 
                                             max_value=len(upload_df)-1, 
                                             value=0,
                                             step=1)
                    
                    selected_row = upload_df.iloc[row_index]
                else:
                    selected_row = upload_df.iloc[0]
                
                # Daten aus der ausgewählten Zeile extrahieren
                input_data = {}
                for feature in all_features:
                    input_data[feature] = selected_row[feature]
                
                actual_quality = selected_row['Qualität_Numerisch']
                
                # Ursachenanalyse durchführen
                if st.button("Ursachenanalyse starten"):
                    st.subheader("Analyse-Ergebnisse")
                    
                    # Farbcodierung für die Qualität
                    if actual_quality >= 8.0:
                        color = "green"
                        quality_text = "Ausgezeichnet"
                    elif actual_quality >= 6.5:
                        color = "orange"
                        quality_text = "Akzeptabel"
                    else:
                        color = "red"
                        quality_text = "Mangelhaft"
                    
                    st.markdown(f"**Aktuelle Qualität:** <span style='color:{color}'>{actual_quality:.2f} ({quality_text})</span>", unsafe_allow_html=True)
                    
                    # Durchführen der Ursachenanalyse
                    analysis = analyze_quality_issues(input_data, actual_quality)
                    
                    if len(analysis['improvements']) > 0:
                        expected_quality = analysis['expected_quality']
                        
                        # Anzeigen der erwarteten Qualität nach Optimierung
                        if expected_quality >= 8.0:
                            expected_color = "green"
                            expected_text = "Ausgezeichnet"
                        elif expected_quality >= 6.5:
                            expected_color = "orange"
                            expected_text = "Akzeptabel"
                        else:
                            expected_color = "red"
                            expected_text = "Mangelhaft"
                        
                        st.markdown(f"**Erwartete Qualität nach Optimierung:** <span style='color:{expected_color}'>{expected_quality:.2f} ({expected_text})</span>", unsafe_allow_html=True)
                        
                        # Anzeigen der empfohlenen Änderungen
                        st.subheader("Empfohlene Parameteränderungen")
                        
                        improvement_df = pd.DataFrame(analysis['improvements'])
                        improvement_df.columns = ['Parameter', 'Aktueller Wert', 'Optimaler Wert', 
                                               'Qualitätsgewinn', '% Änderung']
                        
                        # Formatieren der numerischen Werte
                        improvement_df['% Änderung'] = improvement_df['% Änderung'].map('{:.1f}%'.format)
                        improvement_df['Qualitätsgewinn'] = improvement_df['Qualitätsgewinn'].map('{:.3f}'.format)
                        
                        st.dataframe(improvement_df)
                        
                        # Visualisierung der Qualitätsgewinne
                        if len(improvement_df) > 1:
                            fig = px.bar(improvement_df, 
                                       x='Parameter', 
                                       y='Qualitätsgewinn',
                                       title='Erwarteter Qualitätsgewinn durch Parameteroptimierung')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detaillierte Analyse für den wichtigsten Parameter
                        top_param = analysis['improvements'][0]['parameter']
                        param_idx = treatments.index(top_param)
                        
                        st.subheader(f"Detailanalyse für {top_param}")
                        st.write(f"Der Parameter mit dem größten Einfluss auf die Qualität ist {top_param}.")
                        
                        fig = generate_cf_plot(input_data, param_idx)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Keine signifikanten Verbesserungsmöglichkeiten gefunden.")
                        
    else:  # Manuelle Eingabe
        st.subheader("Prozessparameter eingeben")
        
        # Zweispaltig
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            st.markdown("**Hauptparameter:**")
            for t in treatments:
                min_val = data[t].min()
                max_val = data[t].max()
                mean_val = data[t].mean()
                
                input_data[t] = st.slider(
                    f"{t}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=float((max_val - min_val) / 100),
                    key=f"analysis_{t}"
                )
        
        with col2:
            st.markdown("**Umgebungs- und Produktparameter:**")
            for f in features:
                min_val = data[f].min()
                max_val = data[f].max()
                mean_val = data[f].mean()
                
                input_data[f] = st.slider(
                    f"{f}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=float((max_val - min_val) / 100),
                    key=f"analysis_{f}"
                )
        
        # Eingabe der tatsächlichen Qualität
        st.subheader("Tatsächliche Qualität")
        actual_quality = st.slider(
            "Gemessene Qualität",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            key="measured_slider"
        )
        
        # Ursachenanalyse durchführen
        if st.button("Ursachenanalyse starten", key="manual_analysis"):
            st.subheader("Analyse-Ergebnisse")
            
            # Farbcodierung für die Qualität
            if actual_quality >= 8.0:
                color = "green"
                quality_text = "Ausgezeichnet"
            elif actual_quality >= 6.5:
                color = "orange"
                quality_text = "Akzeptabel"
            else:
                color = "red"
                quality_text = "Mangelhaft"
            
            st.markdown(f"**Aktuelle Qualität:** <span style='color:{color}'>{actual_quality:.2f} ({quality_text})</span>", unsafe_allow_html=True)
            
            # Durchführen der Ursachenanalyse
            analysis = analyze_quality_issues(input_data, actual_quality)
            
            if len(analysis['improvements']) > 0:
                expected_quality = analysis['expected_quality']
                
                # Anzeigen der erwarteten Qualität nach Optimierung
                if expected_quality >= 8.0:
                    expected_color = "green"
                    expected_text = "Ausgezeichnet"
                elif expected_quality >= 6.5:
                    expected_color = "orange"
                    expected_text = "Akzeptabel"
                else:
                    expected_color = "red"
                    expected_text = "Mangelhaft"
                
                st.markdown(f"**Erwartete Qualität nach Optimierung:** <span style='color:{expected_color}'>{expected_quality:.2f} ({expected_text})</span>", unsafe_allow_html=True)
                
                # Anzeigen der empfohlenen Änderungen
                st.subheader("Empfohlene Parameteränderungen")
                
                improvement_df = pd.DataFrame(analysis['improvements'])
                improvement_df.columns = ['Parameter', 'Aktueller Wert', 'Optimaler Wert', 
                                       'Qualitätsgewinn', '% Änderung']
                
                # Formatieren der numerischen Werte
                improvement_df['% Änderung'] = improvement_df['% Änderung'].map('{:.1f}%'.format)
                improvement_df['Qualitätsgewinn'] = improvement_df['Qualitätsgewinn'].map('{:.3f}'.format)
                
                st.dataframe(improvement_df)
                
                # Visualisierung der Qualitätsgewinne
                if len(improvement_df) > 1:
                    fig = px.bar(improvement_df, 
                               x='Parameter', 
                               y='Qualitätsgewinn',
                               title='Erwarteter Qualitätsgewinn durch Parameteroptimierung')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detaillierte Analyse für den wichtigsten Parameter
                top_param = analysis['improvements'][0]['parameter']
                param_idx = treatments.index(top_param)
                
                st.subheader(f"Detailanalyse für {top_param}")
                st.write(f"Der Parameter mit dem größten Einfluss auf die Qualität ist {top_param}.")
                
                fig = generate_cf_plot(input_data, param_idx)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Keine signifikanten Verbesserungsmöglichkeiten gefunden.")

elif page == "Datenexploration":
    st.header("Datenexploration")
    st.write("Analysieren Sie die Zusammenhänge zwischen Prozessparametern und der Qualität.")
    
    # Korrelationsmatrix
    st.subheader("Korrelationsmatrix")
    
    # Berechnen der Korrelationen
    outcome = "Qualität_Numerisch"
    corr = data[treatments + [outcome]].corr()
    
    # Heatmap mit Plotly
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Korrelation zwischen Parametern und Qualität"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance aus dem RandomForest-Modell
    st.subheader("Feature Importance")
    
    # Feature Importance extrahieren
    importances = rf_model.feature_importances_
    feature_names = all_features
    
    # Sortieren
    indices = np.argsort(importances)
    top_k = 15  # Top k Features
    top_indices = indices[-top_k:][::-1]
    
    # Dataframe für die Darstellung
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_indices],
        'Importance': importances[top_indices]
    })
    
    # Balkendiagramm
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_k} wichtigste Features für die Qualitätsvorhersage'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Streudiagramm für ausgewählte Parameter
    st.subheader("Zusammenhang zwischen Parametern und Qualität")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_param = st.selectbox(
            "X-Achse Parameter:",
            treatments,
            index=0,
            key="x_ax_param"
        )
    
    with col2:
        color_by = st.selectbox(
            "Einfärben nach:",
            ["Qualität_Numerisch"] + [t for t in treatments if t != x_param],
            index=0,
            key="die_after"
        )
    
    # Streudiagramm erstellen
    fig = px.scatter(
        data,
        x=x_param,
        y=outcome,
        color=color_by,
        title=f'Zusammenhang zwischen {x_param} und {outcome}',
        color_continuous_scale='viridis',
        opacity=0.7
    )
    
    # Trendlinie hinzufügen
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D-Scatter-Plot für tiefere Analyse
    st.subheader("3D-Analyse")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_param_3d = st.selectbox(
            "X-Achse:",
            treatments,
            index=0,
            key="3d_x"
        )
    
    with col2:
        y_param_3d = st.selectbox(
            "Y-Achse:",
            [t for t in treatments if t != x_param_3d],
            index=0,
            key="3d_y"
        )
    
    with col3:
        z_param = st.selectbox(
            "Z-Achse:",
            ["Qualität_Numerisch"] + [t for t in treatments if t not in [x_param_3d, y_param_3d]],
            index=0,
            key="3d_z"
        )
    
    # 3D-Streudiagramm erstellen
    fig = px.scatter_3d(
        data,
        x=x_param_3d,
        y=y_param_3d,
        z=z_param,
        color='Qualität_Numerisch',
        title=f'3D-Visualisierung: {x_param_3d} vs {y_param_3d} vs {z_param}',
        color_continuous_scale='viridis',
        opacity=0.7
    )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

# Startskript für das Dashboard
if __name__ == '__main__':
    pass  # Platzhalter für die Ausführung des Skripts
    # in console einfach -> streamlit run dashboard.py
