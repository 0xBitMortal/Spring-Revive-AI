# SpringRevive AI

**SIH26240 · AI-Based Spring Revival and Recharge Planning for Tribal Areas**  
Ministry of Tribal Affairs · Government of India  
Deployed Platform: **IBM LinuxONE (s390x)** Enterprise Architecture

---

## 🌟 Overview

SpringRevive AI is a decision-support and groundwater springshed planning platform built for the **Ministry of Tribal Affairs**. It transforms raw geospatial, hydrogeological, and meteorological data into actionable civil engineering blueprints and community advisories for remote tribal areas.

Deployed on **IBM LinuxONE (s390x)**, the platform pairs physics-constrained Machine Learning with ground-level government scheme execution (MGNREGA) and Traditional Ecological Knowledge (TEK).

---

## 🚀 Key Features

### 1. 👷 Human-Centric & Ground Reality
- **Shramik-DPR (MGNREGA Engine):** Automatically generates ready-to-sanction Detailed Project Reports (DPRs) specifying Staggered Contour Trenches (SCT), Loose Boulder Check Dams (LBCD), earthwork excavation ($m^3$), person-days of unskilled labor, and budgets under the statutory 60:40 MGNREGA rule.
- **Vani-Jal (Multilingual Audio Advisory):** Zero-bandwidth, browser-native spoken audio advisories in Hindi and English with intuitive Visual Water Cards for tribal villagers and field teams.
- **Jal-Smriti (Sacred Grove & TEK Layer):** Maps sacred groves (*Sarna, Devrai, Jaher Than*) and computes a **TEK Harmony Index** showing convergence between ancestral wisdom and satellite AI.

### 2. 🔬 High Reliability & Scientific Rigor
- **Physics-Guided Water-Balance Gatekeeper:** Enforces mass-balance conservation ($P = Q + AET + Recharge + \Delta S$) via interactive Plotly Sankey diagrams to prevent "hallucinated" recharge rates.
- **Epistemic Uncertainty Quantification:** Computes variance across all 200 Random Forest decision trees ($\sigma$) to report mathematical confidence intervals and flag sparse data sites.
- **White-Box Explainability (XAI):** Local factor attribution waterfall charts show exactly how slope, infiltration, geology, and rainfall contribute to each spring's score.

### 3. 🌍 Global Reach & Economic Sustainability
- **Blue-Carbon & Green Credit Monetizer:** Estimates annual carbon sequestration (tCO₂e/yr) and community revenue under India's Green Credit Programme (MoEFCC) to fund long-term maintenance.
- **Universal Springshed Typology Adapter:** Dynamically adapts domain weights and slope thresholds for Central Indian Craton, Himalayan Karst, Western Ghats, East African Rift, and Andean Highlands.

---

## 💻 Tech Stack

- **Frontend & App Framework:** Python, Streamlit, HTML5 Web Speech API
- **Machine Learning:** Scikit-Learn (Random Forest Ensemble, Variance Quantification)
- **Geospatial & Visualization:** Folium, Leaflet, Plotly (Waterfall, Sankey, Histogram, Scatter)
- **Enterprise Deployment:** IBM LinuxONE (s390x) High-Reliability Architecture

---

## 🛠️ How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit Web App:**
   ```bash
   streamlit run app.py
   ```
   *(or `python -m streamlit run app.py`)*

3. **Open Browser:**
   Navigate to `http://localhost:8501`.

---

## 📋 CSV Input Schema

Upload a custom CSV with the following mandatory headers:

| Column | Description |
|---|---|
| `spring_id` | Unique identifier (e.g. `SPR-001`) |
| `village` | Gram Panchayat / Village name |
| `latitude` | Decimal degrees (-90 to 90) |
| `longitude` | Decimal degrees (-180 to 180) |
| `rainfall_mm` | Annual precipitation (mm) |
| `slope_deg` | Topographical slope (0–90°) |
| `infiltration_mm_hr` | Soil infiltration rate (mm/hr) |
| `geology` | Lithology (`Fractured`, `Moderate`, `Low permeability`, `Unknown`) |
| `forest_cover_pct` | Catchment forest canopy cover (0–100%) |
| `flow_decline_pct` | Reported discharge decline rate (0–100%) |

---

## 🏛️ License & Scope

Developed for Smart India Hackathon (SIH26240) and Ministry of Tribal Affairs. Demonstrates indicative screening models; all proposed civil works require field hydrogeological validation.
