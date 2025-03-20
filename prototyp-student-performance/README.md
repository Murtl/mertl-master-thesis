# README: CausalAI Experiment on Student Performance Factors
It is important to notice that this project is a prototype and not everything works here. It served as an entry point 
into the topic and helped me a lot. Take a look at the `prototyp-machine-data` folder for a more advanced prototyp
that is closer to the desired result of my master thesis.

## Overview
This project explores the causal relationships between various student-related factors and their impact on **Exam_Score** using **CausalAI**. 
The analysis is based on the Kaggle dataset **"Student Performance Factors"** (synthetic data), examining key influences like Hours_Studied, 
Attendance, Parental_Education_Level, Extracurricular_Activities and so on.

The study was conducted in **four main stages**:
1. **Data Understanding** – Exploring the dataset.
2. **Data Preprocessing** – Cleaning and encoding data for causal analysis.
3. **Causal Model Construction** – Building a Directed Acyclic Graph (DAG) and estimating causal effects.
4. **Causal Model Construction using different Causal Discovery Algorithms** - Automatically generating a Directed Acyclic Graph (DAG) and estimating causal effts.

---

## 1. Data Understanding
**Goal:** Gain an initial understanding of the dataset structure and key attributes.

### Key Steps:
- Loaded the dataset and examined its **shape** and **columns**.
- Identified **numerical** and **categorical** features.
- Generated **descriptive statistics** to understand distributions and value ranges.

### Findings:
- The dataset contains multiple factors affecting student exam performance.
- Several categorical variables require encoding before analysis.

---

## 2. Data Preprocessing
**Goal:** Prepare the dataset for causal inference by handling missing values and encoding categorical variables.

### Key Steps:
- **Missing Values:** Visualized and removed missing data.
- **Categorical Encoding:** Converted categorical variables into numerical representations.
- **Correlation Analysis:** Generated heatmaps to understand feature relationships.

### Findings:
- No significant missing values after preprocessing.
- Some variables, like Parental_Ivolvement and Peer_Influence, showed moderate correlation with exam scores.
- Encoding categorical variables allowed the dataset to be used in causal modeling.

---

## 3. Causal Model Construction
**Goal:** Establish causal relationships and estimate causal effects on exam performance.

### Key Steps:
1. **Directed Acyclic Graph (DAG):**
   - Constructed a DAG to represent causal relationships.
   - Key causal factors identified:
     - **Direct Effects:** Hours_Studied, Attendance, Previous_Scores.
     - **Indirect Effects:** Parental_Education_Level, Extracurricular_Activities.
   
2. **Back-Door Criterion Analysis:**
   - Identified confounding variables (e.g., Parental_Education_Level, Attendance, Previous_Scores).
   - Ensured valid adjustment sets for estimating causal effects.

3. **Causal Effect Estimation:**
   - Used **regression adjustment** and **Inverse Probability Weighting (IPW)**.
   - Validated results with **placebo tests** and **bootstrap resampling**.

4. **Counterfactual & Policy Simulations:**
   - Estimated individual treatment effects (ITE) for personalized recommendations.
   - Simulated policy interventions (e.g., increasing study hours by 20).
   - Analyzed hypothetical scenarios (e.g., effect of 5 additional study hours).

### Findings:
- **Hours_Studied has a strong causal impact** on exam scores.
- **Parental_Education_Level influences Hours_Studied and Attendance**, indirectly affecting performance.
- **Extracurricular_Activities have an indirect impact** through study habits.
- **Inverse Probability Weighting confirmed a significant causal effect**, controlling for confounders.
- **Refutation tests confirmed model robustness**, ensuring reliability.

---

## Implications for CausalAI Usage
### Advantages:
✅ Identifies true **causal** relationships, not just correlations.
✅ Enables **counterfactual reasoning** (e.g., "What if the student had studied 5 more hours?").
✅ Supports **policy simulations** for real-world interventions.
✅ Provides **interpretable results** for decision-making in education.

### Limitations:
⚠ Requires **domain knowledge** to define valid causal relationships.
⚠ Assumes **no unmeasured confounders** – missing variables could introduce bias.
⚠ Relies on **assumption validation** through refutation tests.

---

## Usage
### 1. Estimate Individual Treatment Effects (ITE)
Predict how much an individual’s exam score would improve if they studied more:
```python
ite_prediction = model.estimate_individual_treatment_effect(student_id)
print(f"Expected score improvement: {ite_prediction}")
```

### 2. Policy Simulations
Simulate the impact of increasing study hours:
```python
df_encoded['Hours_Studied_new'] = df_encoded['Hours_Studied'] + 20
df_encoded['predicted_score_new'] = causal_model.predict(df_encoded['Hours_Studied_new'])
```

### 3. Counterfactual Analysis
Estimate what would have happened if a student had studied 5 more hours:
```python
counterfactual_score = causal_model.estimate_counterfactual(student_id, hours_studied + 5)
```

## 4. Causal Model Construction using Causal Discovery Algorithms
**Goal:** Establish causal relationships and estimate causal effects on exam performance using automatic causal discovery algorithms.

### Key Steps:
1. **Directed Acyclic Graph (DAG) using PC-Algorithm**

2. **Directed Acyclic Graph (DAG) using GES-Algorithm:**

3. **Directed Acyclic Graph (DAG) using LINGAM-Algorithm:**
   
4. **Causal Effect Estimation:**
   - Used identification, estimation and refute_estimation to test the automatically generated DAGs.
  
### Findings:
- **PC** algorithm archieved the best DAG.
- **Automatic Causal Discovery** can be used to easily create DAGs -> but these must be evaluated by humans with domain knowledge.

---

## Application in Manufacturing (e.g., Food Packaging Industry)
### How CausalAI Can Benefit Machine Manufacturers
CausalAI is not limited to education—it can be highly beneficial for **manufacturers**, particularly those producing large machines for **food packaging**. By applying causal analysis, companies can:

- **Optimize Machine Performance**: Identify the causal impact of machine settings (e.g., temperature, speed, pressure) on packaging quality and efficiency.
- **Reduce Downtime**: Analyze how different factors (e.g., maintenance schedules, operator expertise, environmental conditions) influence machine failures and optimize preventive maintenance.
- **Improve Production Yield**: Estimate the causal effects of raw material quality, processing speed, and operational workflows on defect rates.
- **Enhance Energy Efficiency**: Determine the causal factors affecting energy consumption and implement adjustments to reduce costs.
- **Predict and Prevent Failures**: Use counterfactual analysis to estimate potential failures before they occur and take proactive actions.

### Example Use Case
If a manufacturer observes inconsistent packaging quality, a causal model could analyze:
- **Direct Effects:** Machine speed, sealing temperature, material thickness.
- **Indirect Effects:** Operator experience, maintenance history, room humidity.

By running causal inference, the company can **identify the root cause** (e.g., "Sealing temperature has a direct effect on defect rates") and make data-driven decisions to **improve quality and efficiency**.

### Conclusion
By leveraging CausalAI, **machine manufacturers** can move beyond traditional correlation-based analytics and gain deeper, actionable insights to **optimize performance, reduce costs, and enhance reliability** in food packaging and beyond.

