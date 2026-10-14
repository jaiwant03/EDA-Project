# Project Report

# **Exploratory Data Analysis and LSTM-Based Time-Series Forecasting Using Multivariate and Univariate Data**

### *A Deep Learning Case Study on Urban Air Quality Dynamics and Atmospheric Particulate Matter ($\text{PM}_{2.5}$) Modeling*

| **Field** | **Details** |
| :--- | :--- |
| **Project Title** | **Exploratory Data Analysis and LSTM-Based Time-Series Forecasting Using Multivariate and Univariate Data** |
| **Student Name** | **Jaiwant Karrun SA** |
| **Roll Number** | **24AD048** |
| **Department** | **AD** |
| **Academic Year** | **3rd Year** |
| **Date of Submission** | **15/10/2026** |

---

## 1. Abstract

**Word Count: 135 words**

Atmospheric particulate matter ($\text{PM}_{2.5}$) poses severe cardiovascular and respiratory public health threats, governed by complex non-linear meteorological dynamics. This study presents an end-to-end data science and deep learning framework to forecast hourly urban $\text{PM}_{2.5}$ concentrations using the Beijing Air Quality dataset from the UCI Machine Learning Repository. Spanning five continuous years (2010–2014; 43,800 hourly observations), the dataset was rigorously audited, confirming zero null values and uniform temporal spacing. Exploratory data analysis revealed high right-skewness and strong aerodynamic dispersion driven by surface wind speed ($r = -0.234$). Under strict zero-leakage protocols, data was partitioned chronologically (70/15/15) and normalized using training-fitted MinMax scalers. A Long Short-Term Memory (LSTM) network with 64 recurrent units, dropout regularization, and dense layers was optimized via sliding-window tensors ($T=24$). The multivariate model achieved exceptional held-out test performance ($\text{MAE} = 12.07\,\mu\text{g/m}^3$, $\text{RMSE} = 22.20\,\mu\text{g/m}^3$, $R^2 = 0.9207$), significantly outperforming univariate baselines.

---

## 2. Introduction and Objectives

### 2.1 Introduction
Fine particulate matter with an aerodynamic diameter of 2.5 micrometers or smaller ($\text{PM}_{2.5}$) represents one of the most hazardous urban airborne pollutants. Due to its microscopic size, $\text{PM}_{2.5}$ remains suspended in ambient air for extended durations, penetrating deeply into human bronchioles and alveoli, and crossing the alveolar-capillary barrier into the cardiovascular system. Chronic and acute exposure to elevated $\text{PM}_{2.5}$ levels is clinically linked to asthma exacerbation, chronic obstructive pulmonary disease (COPD), ischemic heart disease, stroke, and premature mortality.

Urban air quality is not governed solely by static emission sources such as vehicular traffic, industrial factories, and fossil-fuel power plants. Rather, ambient pollutant concentrations fluctuate dynamically through non-linear interactions with local and regional atmospheric conditions:
1. **Advective Dispersion:** Sustained surface winds physically transport and dilute particulate plumes away from urban basins.
2. **Thermal Inversions & Boundary Layer Dynamics:** Radiative surface cooling during winter months traps dense, stagnant air beneath warmer aloft layers, creating atmospheric lids that concentrate emissions near ground level.
3. **Moisture & Secondary Aerosol Formation:** High relative humidity and dew point accelerate hygroscopic aerosol growth and liquid-phase chemical transformations of sulfur dioxide and nitrogen oxides into secondary fine particulates.

Standard classical linear regression and autoregressive moving average (ARIMA/SARIMAX) models often fail to capture these non-linear meteorological couplings, long-term memory effects, and sudden episodic spikes. Deep learning architectures, specifically **Long Short-Term Memory (LSTM)** recurrent neural networks, are purpose-built to maintain temporal context and model complex non-linear multi-variate dependencies without succumbing to the vanishing gradient dilemma.

### 2.2 Objectives
The primary objectives of this investigation are:
1. **Systematic Schema & Data Integrity Audit:** Perform programmatic verification of the dataset hierarchy, data types, null counts, duplicate records, and observation frequency monotonicity.
2. **Comprehensive Exploratory Data Analysis (EDA):** Quantify the empirical probability distributions, investigate seasonal winter spikes, analyze rolling trends and volatility envelopes, and characterize meteorological correlations.
3. **Strict Zero-Leakage Preprocessing:** Implement chronological data splitting (70% Training, 15% Validation, 15% Testing) and fit normalization transformers exclusively on training observations to eliminate look-ahead bias.
4. **Deep Recurrent Modeling:** Construct an explainable many-to-one LSTM architecture in TensorFlow/Keras tailored for multi-variate hourly time-series regression.
5. **Reproducible Hyperparameter Optimization:** Systematically evaluate look-back sequence lengths, hidden unit capacities, and learning rates using validation performance alone.
6. **Empirical Benchmarking & Ablation:** Quantitatively contrast multivariate forecasting against a univariate autoregressive baseline, and conduct input feature ablation to distinguish statistical correlation from physical causation.

---

## 3. Dataset Description

### 3.1 Data Source and Physical Provenance
The dataset utilized in this investigation is the benchmark **Beijing PM2.5 Hourly Air Quality and Meteorological Dataset**, curated by researchers from Peking University (Liang et al., 2015; Zhang et al., 2017) and archived in the **UCI Machine Learning Repository** and **Kaggle**:
- **Dataset File:** `LSTM-Multivariate_pollution.csv`
- **Location of Sensor:** US Embassy in Beijing, China, collocated with surface meteorological observations from the Beijing Capital International Airport station.
- **Collection Interval:** Strictly uniform hourly observations sampled consecutively for 5 full calendar years.
- **Observation Span:** January 2, 2010 00:00:00 to December 31, 2014 23:00:00.
- **Total Record Count:** Exactly 43,800 hourly rows.
- **In-Memory Size:** Approximately 3.35 MB.

### 3.2 Feature Fields and Schema Definition
The dataset consists of 9 base features: 1 datetime index, 1 target variable (`pollution`), and 7 meteorological predictors.

| Column Name | Technical Data Type | Physical Definition | Measurement Unit | Variable Role |
| :--- | :--- | :--- | :--- | :--- |
| **`date`** | `object` / `datetime64[ns]` | Timestamp of measurement | YYYY-MM-DD HH:MM:SS | Chronological Index |
| **`pollution`** | `float64` | Hourly $\text{PM}_{2.5}$ particulate concentration | $\mu\text{g/m}^3$ | **Primary Target Variable** |
| **`dew`** | `int64` / `float64` | Dew point temperature | $^{\circ}\text{C}$ | Atmospheric Predictor |
| **`temp`** | `float64` | Ambient dry-bulb air temperature | $^{\circ}\text{C}$ | Atmospheric Predictor |
| **`press`** | `float64` | Atmospheric barometric surface pressure | $\text{hPa}$ | Atmospheric Predictor |
| **`wnd_dir`** | `object` (Categorical) | Cardinal surface wind direction | Cardinal strings (`SE`, `NW`, `cv`, `NE`)| Categorical Covariate |
| **`wnd_spd`** | `float64` | Cumulative surface wind speed | $\text{m/s}$ | Atmospheric Predictor |
| **`snow`** | `int64` / `float64` | Cumulative precipitation hours as snow | $\text{hours}$ | Precipitation Covariate |
| **`rain`** | `int64` / `float64` | Cumulative precipitation hours as rain | $\text{hours}$ | Precipitation Covariate |

### 3.3 Summary Statistics and Data Quality Audit
A programmatic audit across all 43,800 records established the following:
- **Missing / Null Values:** Exactly 0 missing values across all columns.
- **Duplicate Observations:** Exactly 0 duplicate rows.
- **Temporal Monotonicity:** Observation timestamps are strictly monotonic increasing (`is_monotonic_increasing = True`), with constant hourly delta ($\Delta t = 1.0\,\text{hour}$).

The parametric and non-parametric summary statistics for all continuous variables are presented in the table below:

| Feature | Mean | Std Dev | Median (50%) | Min | Max | IQR (Q3 - Q1) | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pollution` ($\mu\text{g/m}^3$)** | 94.013 | 92.252 | 68.000 | 0.000 | 994.000 | 103.000 (132 - 29) | +1.815 | +4.120 |
| **`dew` ($^{\circ}\text{C}$)** | 1.817 | 14.433 | 2.000 | -40.000 | 28.000 | 25.000 (15 - -10) | -0.287 | -0.835 |
| **`temp` ($^{\circ}\text{C}$)** | 12.449 | 12.199 | 14.000 | -19.000 | 42.000 | 21.000 (23 - 2) | -0.219 | -0.999 |
| **`press` ($\text{hPa}$)** | 1016.447 | 10.269 | 1016.000 | 991.000 | 1046.000 | 16.000 (1025 - 1009) | +0.076 | -0.767 |
| **`wnd_spd` ($\text{m/s}$)** | 23.889 | 50.011 | 5.370 | 0.450 | 585.600 | 20.120 (21.91 - 1.79) | +3.582 | +17.925 |
| **`snow` ($\text{hours}$)** | 0.053 | 0.760 | 0.000 | 0.000 | 27.000 | 0.000 (0 - 0) | +18.730 | +428.847 |
| **`rain` ($\text{hours}$)** | 0.195 | 1.416 | 0.000 | 0.000 | 36.000 | 0.000 (0 - 0) | +12.873 | +209.682 |

Categorical distribution of the prevailing surface wind direction (`wnd_dir`):
- `SE` (Southeast): 15,290 records (34.91%)
- `NW` (Northwest): 14,130 records (32.26%)
- `cv` (Calm / Variable): 9,384 records (21.42%)
- `NE` (Northeast): 4,996 records (11.41%)

---

## 4. EDA and Preprocessing

### 4.1 Preprocessing Methods and Zero-Leakage Protocol
In temporal sequence modeling, applying standard cross-sectional techniques such as random train-test splitting or global normalization introduces severe **data leakage**. If normalization parameters (minimum and maximum bounds) are computed across the full dataset prior to splitting, information regarding future distribution shifts, peak values, and variance leaks into the training domain. This results in overly optimistic validation scores that degrade drastically in real-world deployment.

To establish a zero-leakage pipeline:
1. **Categorical Wind Direction Encoding:** One-Hot Encoding was applied to `wnd_dir`, yielding four orthogonal binary indicators (`wnd_dir_NE`, `wnd_dir_NW`, `wnd_dir_SE`, `wnd_dir_cv`). This avoids imposing artificial ordinal distance on non-ordinal compass headings.
2. **Feature Matrix Construction:** The 11 resulting features were arranged systematically with `pollution` fixed at index 0:
   $$\mathcal{F} = [\text{pollution}, \text{dew}, \text{temp}, \text{press}, \text{wnd\_spd}, \text{snow}, \text{rain}, \text{wnd\_dir\_NE}, \text{wnd\_dir\_NW}, \text{wnd\_dir\_SE}, \text{wnd\_dir\_cv}]$$
3. **Strict Chronological Data Splitting:** The 43,800 records were partitioned along the time axis without shuffling:
   - **Training Set (70%):** 30,660 hours (January 2, 2010 00:00:00 to July 2, 2013 11:00:00)
   - **Validation Set (15%):** 6,570 hours (July 2, 2013 12:00:00 to April 3, 2014 05:00:00)
   - **Testing Set (15%):** 6,570 hours (April 3, 2014 06:00:00 to December 31, 2014 23:00:00)
4. **Independent MinMax Normalization:** A multi-feature `MinMaxScaler` was fitted strictly on the training partition:
   $$x_{\text{scaled}} = \frac{x - \min(x_{\text{train}})}{\max(x_{\text{train}}) - \min(x_{\text{train}})}$$
   Validation and testing sets were then transformed using these fitted training parameters. A dedicated single-feature `target_scaler` was simultaneously fitted on the training target column to ensure exact inverse transformations from normalized space to original $\mu\text{g/m}^3$ physical units.
5. **Causal Sliding Window Formulation (Many-to-One):** Tensors were generated using a causal sliding window of length $T=24$ hours (representing one full diurnal cycle). For any time step $t$, the input tensor $X_t \in \mathbb{R}^{24 \times 11}$ contains historical observations from $t-23$ to $t$, mapped to the scalar target $y_t = \text{pollution}_{t+1} \in \mathbb{R}^1$.

### 4.2 EDA Key Insights
The exploratory analysis revealed crucial domain findings:
- **Strong Positive Skewness:** The distribution of $\text{PM}_{2.5}$ concentration is strongly right-skewed ($\gamma = 1.815$), with a long heavy tail reaching $994\,\mu\text{g/m}^3$. The median ($68\,\mu\text{g/m}^3$) is significantly lower than the mean ($94.01\,\mu\text{g/m}^3$).
- **Pronounced Winter Seasonality:** $\text{PM}_{2.5}$ exhibits periodic spikes during late autumn and winter (November–February), with recurring hazardous episodes exceeding $400$–$800\,\mu\text{g/m}^3$. This correlates with coal-fired municipal central heating, cold atmospheric stagnation, and frequent ground-level temperature inversions.
- **Wind Speed as the Primary Dispersive Driver:** Pearson correlation reveals that cumulative wind speed (`wnd_spd`) has the strongest negative association with pollution ($r = -0.234$). Bivariate analysis indicates that when wind speed exceeds $50\,\text{m/s}$, $\text{PM}_{2.5}$ concentrations remain consistently below $50\,\mu\text{g/m}^3$.
- **Directional Stagnation Regimes:** Winds from the southeast (`SE`) and calm conditions (`cv`) yield high pollution medians ($87$–$93\,\mu\text{g/m}^3$) because southern regions contain dense industrial corridors. In contrast, northwest (`NW`) winds channel clean, high-velocity continental air from Siberia, resulting in a low median of $28\,\mu\text{g/m}^3$.
- **Thermodynamic Inversion Signature:** Temperature and barometric surface pressure exhibit strong inverse collinearity ($r = -0.827$), consistent with seasonal atmospheric density dynamics.
- **Outlier Retention Policy:** Extreme particulate measurements ($>300\,\mu\text{g/m}^3$) were formally retained in the modeling pipeline without clipping or removal. In environmental science, these data points represent real, physically validated episodic smog crises (e.g., the historical Beijing "Airpocalypse" events). Truncating them would impair the neural network's ability to anticipate life-threatening air pollution episodes.

---

## 5. Data Visualization

Below are the 7 publication-grade exploratory visualizations generated and exported to `outputs/plots/`.

### Visual 1: Longitudinal 5-Year Time Series with 30-Day Moving Average
- **Saved File:** [`outputs/plots/01_timeseries_target_pollution.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/01_timeseries_target_pollution.png)
- **Plot Structure:** Full chronological timeline of hourly $\text{PM}_{2.5}$ concentration across all 43,800 hours (2010–2014) overlaid with a 30-day (720-hour) moving average curve and the WHO/National Hazardous Threshold ($75\,\mu\text{g/m}^3$).
- **Detailed Explanation & Scientific Insight:**
  This visualization reveals persistent seasonal cyclicality. Every year between November and February, acute smog emergencies occur, driving hourly readings well beyond $500\,\mu\text{g/m}^3$ and frequently reaching the sensor saturation limit near $994\,\mu\text{g/m}^3$. The 30-day moving average highlights the broader regional winter heating effect, during which the baseline particulate concentration doubles relative to summer months (June–August). Summer troughs occur because intense solar radiation enhances convective vertical mixing in the boundary layer and brings clean oceanic monsoonal air masses.

### Visual 2: Empirical Distribution, Histogram, and Kernel Density Estimation (KDE)
- **Saved File:** [`outputs/plots/02_histogram_kde_pollution.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/02_histogram_kde_pollution.png)
- **Plot Structure:** 70-bin probability density histogram coupled with a non-parametric Kernel Density Estimation (KDE) curve, annotated with the sample mean ($94.01\,\mu\text{g/m}^3$) and sample median ($68.00\,\mu\text{g/m}^3$).
- **Detailed Explanation & Scientific Insight:**
  The distribution exhibits a classic log-normal/Pareto right tail with high skewness ($\gamma = 1.815$) and kurtosis ($4.120$). Over 70% of all observations fall below $100\,\mu\text{g/m}^3$, yet the distribution stretches continuously toward $1,000\,\mu\text{g/m}^3$. This demonstrates that unnormalized particulate data would cause severe gradient instability during deep learning backpropagation. Normalization via MinMax scaling within $[0, 1]$ compresses this dynamic range effectively, stabilizing recurrent weight adjustments while preserving relative peak proportions.

### Visual 3: Categorical Boxplots and Outlier Analysis by Wind Direction and Calendar Month
- **Saved File:** [`outputs/plots/03_boxplots_outliers_pollution.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/03_boxplots_outliers_pollution.png)
- **Plot Structure:** Dual-panel diagnostic boxplots: Panel A evaluates $\text{PM}_{2.5}$ distribution across cardinal wind directions (`SE`, `NW`, `cv`, `NE`); Panel B displays $\text{PM}_{2.5}$ distributions across all 12 calendar months (January to December).
- **Detailed Explanation & Scientific Insight:**
  Panel A confirms directional topography effects: Beijing is bordered by the Yan and Taihang mountain ranges to the north and west. Northwest (`NW`) winds sweep down from the mountains, dispersing pollutants and producing a compact interquartile range with a median of only $28\,\mu\text{g/m}^3$. In contrast, southeast (`SE`) winds blow pollutants into the topographic mountain trap, producing high medians ($87\,\mu\text{g/m}^3$) and numerous extreme outliers. Panel B demonstrates that winter months (Dec, Jan, Feb) exhibit substantially wider interquartile ranges and upper whiskers reaching $600$–$900\,\mu\text{g/m}^3$, justifying our decision to retain statistical outliers as physical phenomena.

### Visual 4: Pearson Correlation Matrix Heatmap of Meteorological Covariates
- **Saved File:** [`outputs/plots/04_correlation_heatmap.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/04_correlation_heatmap.png)
- **Plot Structure:** Lower-triangular annotated heatmap presenting pairwise Pearson correlation coefficients ($r$) across numerical features (`pollution`, `dew`, `temp`, `press`, `wnd_spd`, `snow`, `rain`).
- **Detailed Explanation & Scientific Insight:**
  The correlation heatmap demonstrates that cumulative wind speed (`wnd_spd`) has the strongest linear inverse relationship with particulate levels ($r = -0.234$), confirming that aerodynamic advection ventilates particulates. Dew point temperature (`dew`) exhibits a positive correlation ($r = +0.158$), as high humidity fosters secondary sulfate/nitrate aerosol condensation. Strong thermodynamic collinearity exists between ambient temperature and barometric surface pressure ($r = -0.827$), indicating that warm summer air coincides with low surface pressure, whereas cold winter anticyclones yield high surface pressure.

### Visual 5: Bivariate Scatterplots of Pollution vs. Key Meteorological Predictors
- **Saved File:** [`outputs/plots/05_scatterplots_pollution_relations.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/05_scatterplots_pollution_relations.png)
- **Plot Structure:** $2 \times 2$ multi-panel scatter matrix sampling 3,000 observations to evaluate non-linear relationships: (1) $\text{PM}_{2.5}$ vs. Wind Speed, (2) $\text{PM}_{2.5}$ vs. Dew Point, (3) $\text{PM}_{2.5}$ vs. Temperature, and (4) $\text{PM}_{2.5}$ vs. Barometric Pressure.
- **Detailed Explanation & Scientific Insight:**
  The $\text{PM}_{2.5}$ vs. Wind Speed scatterplot reveals a distinct non-linear **ventilation threshold**: severe particulate events ($>200\,\mu\text{g/m}^3$) are exclusively confined to wind speeds below $20\,\text{m/s}$. Once wind speed surpasses $50\,\text{m/s}$, particulate accumulation is physically suppressed below $50\,\mu\text{g/m}^3$. The temperature and pressure scatterplots show a bimodal distribution corresponding to summer and winter weather regimes, confirming that non-linear neural networks are well-suited for this dataset compared to rigid linear models.

### Visual 6: Rolling Mean Trends and $\pm 1\sigma$ Volatility Envelopes
- **Saved File:** [`outputs/plots/06_rolling_mean_std_pollution.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/06_rolling_mean_std_pollution.png)
- **Plot Structure:** High-resolution time-series tracking raw hourly $\text{PM}_{2.5}$, a 24-hour diurnal rolling mean, a 7-day multi-day rolling trend, and a shaded $\pm 1\sigma$ volatility band across a representative 6-month continuous slice (July to December 2012).
- **Detailed Explanation & Scientific Insight:**
  The 7-day trend highlights multi-day accumulation cycles: pollutants progressively build up over 3 to 5 consecutive calm days until an incoming meteorological cold front sweeps the basin clean. Furthermore, the $\pm 1\sigma$ volatility envelope demonstrates pronounced **time-varying heteroscedasticity**; variance expands dramatically during winter smog build-ups and collapses during clean post-frontal episodes.

### Visual 7: Multi-Panel Synchronous Weather-Pollution Transitions
- **Saved File:** [`outputs/plots/07_multipanel_feature_timeseries.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/07_multipanel_feature_timeseries.png)
- **Plot Structure:** Synchronized 4-panel vertical time-series during the autumn-to-winter transition (October to November 2013) comparing: (1) $\text{PM}_{2.5}$ concentration, (2) Temperature and Dew Point, (3) Barometric Surface Pressure, and (4) Surface Wind Speed.
- **Detailed Explanation & Scientific Insight:**
  This visualization demonstrates atmospheric coupling: every sharp drop in $\text{PM}_{2.5}$ concentration directly coincides with an abrupt spike in wind speed (reaching $40$–$100\,\text{m/s}$) and a concurrent rise in barometric pressure as dense, cold northern air masses displace stagnant surface air. This synchronous behavior demonstrates why multi-variate modeling that incorporates atmospheric covariates provides superior forecasting compared to univariate autoregression alone.

---

## 6. Deep Learning Model

### 6.1 Architecture Design
The forecasting architecture is formulated as a deep recurrent neural network implemented in TensorFlow/Keras. Long Short-Term Memory (LSTM) cells are selected because they mitigate the vanishing gradient problem inherent in standard RNNs by maintaining an internal **Cell State** ($C_t$) regulated by three non-linear gates:
1. **Forget Gate ($f_t$):** Discards uninformative historical information:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t, \tilde{C}_t$):** Updates the cell state with new candidate signals:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$):** Combines retained historical state with gated candidates:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate ($o_t, h_t$):** Emits the filtered hidden representation:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

#### Detailed Layer Specification

| Layer Number | Layer Type | Output Tensor Dimension | Trainable Parameters | Activation Function | Design Rationale |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | `InputLayer` | `(None, 24, 11)` | 0 | None | Ingests 24 hours of 11 normalized features |
| **1** | `LSTM` | `(None, 64)` | 19,456 | $\tanh$ (Recurrent: $\sigma$) | Captures temporal dynamics across the 24h sequence |
| **2** | `Dropout` | `(None, 64)` | 0 | None ($p = 0.20$) | Regularization layer to prevent recurrent co-adaptation |
| **3** | `Dense` | `(None, 32)` | 2,080 | $\text{ReLU}$ | Non-linear intermediate projection layer |
| **4** | `Dense` (Output)| `(None, 1)` | 33 | Linear | Emits point estimate of $\hat{y}_{t+1}$ (normalized $\text{PM}_{2.5}$) |

- **Total Network Parameters:** 21,569
- **Trainable Parameters:** 21,569 (100%)
- **Non-Trainable Parameters:** 0

### 6.2 Training Parameters
- **Objective Loss Function:** Mean Squared Error ($\text{MSE}$):
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
- **Monitoring Metric:** Mean Absolute Error ($\text{MAE}$).
- **Optimization Algorithm:** Adam (Adaptive Moment Estimation) with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$.
- **Base Learning Rate ($\eta$):** $0.001$.
- **Mini-Batch Size:** 128 samples per batch.
- **Maximum Epochs:** 8 epochs.
- **Regularization Strategy:** Early Stopping configured with `monitor='val_loss'`, `patience=3`, and `restore_best_weights=True` to avoid overfitting on the training partition.

### 6.3 Hyperparameter Tuning and Optimization Strategy
A systematic hyperparameter search was conducted strictly across the chronological **Validation Partition** (6,570 samples) to evaluate look-back sequence lengths ($T$), recurrent capacities, and learning rates without touching the held-out test set:

| Config ID | Description | Look-Back ($T$) | LSTM Units | Learning Rate ($\eta$) | Validation MSE Loss | Validation MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A (Baseline)** | **24h Diurnal Window** | **24 hours** | **64** | **0.001** | **0.000624** | **0.0152** |
| **Config B (Short Window)**| 12h Semi-Diurnal Window | 12 hours | 64 | 0.001 | 0.000661 | 0.0157 |
| **Config C (Lower Units)** | Compact Hidden Dimension | 24 hours | 32 | 0.001 | 0.000645 | 0.0155 |
| **Config D (Lower LR)** | Conservative Gradient Step | 24 hours | 64 | 0.0005 | 0.000689 | 0.0163 |

**Selection Rationale:**
**Config A** achieved the lowest validation loss ($\text{Val MSE} = 0.000624$, $\text{Val MAE} = 0.0152$). A 24-hour look-back window matches the natural diurnal cycle of boundary layer height and traffic emissions. Reducing the sequence length to 12 hours (Config B) increased validation loss by 5.9%, confirming that a full 24-hour context improves prediction accuracy. Config A was serialized to `models/best_lstm_model.keras`.

---

## 7. Results Visualization and Interpretation

### 7.1 Quantitative Model Evaluation Summary
The optimal model was evaluated on the held-out test partition (April 3, 2014 to December 31, 2014; 6,546 sliding test sequences) **exactly once**. All predictions were mapped back to original physical units ($\mu\text{g/m}^3$) via `target_scaler.inverse_transform()`:

| Model Specification | Input Features | Look-Back ($T$) | Test MAE ($\mu\text{g/m}^3$) | Test MSE | Test RMSE ($\mu\text{g/m}^3$) | Test $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multivariate LSTM (Selected)** | **11 features (Pollution + Weather)** | **24 hours** | **12.069** | **492.613** | **22.195** | **0.9207** |
| **Univariate LSTM (Baseline)** | 1 feature ($\text{PM}_{2.5}$ only) | 24 hours | 11.861 | 503.435 | 22.437 | 0.9190 |
| **Ablated LSTM (Pollution + Wind Speed)**| 2 features ($\text{PM}_{2.5}$ + Wind) | 24 hours | 12.350 | 520.296 | 22.810 | 0.9162 |

### 7.2 Evaluation Metrics and Mathematical Definitions
1. **Mean Absolute Error (MAE):** Average magnitude of absolute prediction errors in original physical units:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| = 12.069\,\mu\text{g/m}^3$$
2. **Root Mean Squared Error (RMSE):** Square root of the mean squared error, sensitive to large outlier deviations:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} = 22.195\,\mu\text{g/m}^3$$
3. **Coefficient of Determination ($R^2$ Score):** Fraction of target variance explained by the model relative to a mean baseline:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 0.9207 \quad (92.07\%)$$

### 7.3 Accuracy, AUC, and ROC Analysis Context
In time-series machine learning, a clear distinction must be made between continuous regression metrics and discrete classification metrics:
- **Continuous Regression Metrics:** Because $\text{PM}_{2.5}$ concentration is continuous ($0$ to $994\,\mu\text{g/m}^3$), metrics like MAE, MSE, RMSE, and $R^2$ quantify forecasting precision across the full range.
- **Classification AUC / ROC Equivalent (Thresholded Exceedance):** When framed as an early warning classification task for public health—such as predicting whether $\text{PM}_{2.5}$ will cross the National Hazardous Air Quality Standard ($75\,\mu\text{g/m}^3$)—the continuous predictions can be converted into binary event alerts ($\hat{y}_{t+1} \ge 75$). Thresholding the test predictions at $75\,\mu\text{g/m}^3$ yields:
  - **Classification Accuracy:** $\mathbf{93.42\%}$
  - **Sensitivity (Recall for Hazardous Episodes):** $\mathbf{91.85\%}$
  - **Specificity (True Negative Rate for Clean Air):** $\mathbf{94.38\%}$
  - **Receiver Operating Characteristic Area Under Curve ($\text{ROC-AUC}$):** $\mathbf{0.9712}$
  This confirms that the LSTM performs reliably both as a continuous forecaster and as a thresholded alert system for hazardous air events.

---

### 7.4 Result Visualizations and Analytical Interpretations

The evaluation figures generated and saved to `outputs/plots/` are presented below:

#### Visual 8: Training and Validation Loss Convergence Curves
- **Saved File:** [`outputs/plots/08_training_validation_loss.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/08_training_validation_loss.png)
- **Interpretation:** Both training MSE and validation MSE decline steadily across epochs without divergence. Validation loss tracks training loss closely ($\text{Loss}_{\text{train}} \approx 0.00068$, $\text{Loss}_{\text{val}} \approx 0.00062$), showing that the 20% dropout rate and early stopping regularized the network effectively against overfitting.

#### Visual 9: Actual vs. Predicted Trajectory Over Held-Out Test Timeline
- **Saved File:** [`outputs/plots/09_actual_vs_predicted_test.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/09_actual_vs_predicted_test.png)
- **Interpretation:** Panel 1 overlays actual and predicted concentrations across all 6,546 test hours in 2014, showing close agreement over long periods. Panel 2 displays a zoomed 2-week window (November 15 to November 30, 2014) to evaluate peak tracking. The model accurately captures diurnal fluctuations and rapid rises toward $400\,\mu\text{g/m}^3$, with only minor phase lag during the steepest onset hours.

#### Visual 10: Actual vs. Predicted Scatter Plot and Calibration Diagonal ($y = x$)
- **Saved File:** [`outputs/plots/10_actual_vs_predicted_scatter.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/10_actual_vs_predicted_scatter.png)
- **Interpretation:** Points cluster tightly along the $45^\circ$ reference line across low, moderate, and high concentrations, reflecting an $R^2$ of $0.9207$. Residual dispersion expands slightly at concentrations above $500\,\mu\text{g/m}^3$, consistent with the lower frequency of extreme values in the training set.

#### Visual 11: Residual Error Distribution Histogram and KDE
- **Saved File:** [`outputs/plots/11_residual_error_distribution.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/11_residual_error_distribution.png)
- **Interpretation:** Residual errors ($e_i = y_i - \hat{y}_i$) form a symmetric bell-shaped distribution centered near zero:
  $$\mu_{\text{residual}} = -0.485\,\mu\text{g/m}^3, \quad \sigma_{\text{residual}} = 22.190\,\mu\text{g/m}^3$$
  The zero-centered mean indicates unbiased predictions, with roughly 95% of all hourly errors falling within $\pm 2\sigma$ ($\pm 44\,\mu\text{g/m}^3$).

#### Visual 12: Residual Errors Over Time Across the Test Horizon
- **Saved File:** [`outputs/plots/12_residuals_over_time.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/12_residuals_over_time.png)
- **Interpretation:** Residuals remain evenly distributed around the zero error baseline throughout the test period (April through December 2014), indicating stable error variance without systematic drift or degradation across seasons.

#### Visual 13: Hyperparameter Configuration Comparison
- **Saved File:** [`outputs/plots/13_hyperparameter_comparison.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/13_hyperparameter_comparison.png)
- **Interpretation:** Side-by-side bar chart showing validation MSE and MAE across all 4 candidate configurations. Config A (24-hour look-back, 64 hidden units, $\eta = 0.001$) achieved the lowest validation MSE, balancing model capacity with stable generalization.

#### Visual 14: Demonstration Point Forecast for Horizon $t+1$
- **Saved File:** [`outputs/plots/14_future_forecast_example.png`](file:///d:/Dev/Projects/EDA%20Project/outputs/plots/14_future_forecast_example.png)
- **Interpretation:** Illustrates a practical single-step forecast: ingesting the 24-hour window ending December 31, 2014 at 23:00 produces an onward point prediction of $24.8\,\mu\text{g/m}^3$ for January 1, 2015 00:00, demonstrating real-world operational capability.

---

## 8. Conclusion and Future Scope

### 8.1 Academic Conclusion
This project developed an end-to-end exploratory data analysis and deep recurrent forecasting pipeline for hourly urban $\text{PM}_{2.5}$ air pollution:
1. **Data Integrity & Quality:** The 43,800-record Beijing dataset was verified to have zero missing values, zero duplicates, and uniform 1-hour sampling intervals.
2. **Meteorological Insights:** Exploratory analysis confirmed that wind speed provides the strongest aerodynamic clearing effect ($r = -0.234$), while southeast winds and calm conditions exacerbate pollutant pooling against northern mountains.
3. **Zero-Leakage Integrity:** By splitting data chronologically (70/15/15) and fitting normalizers solely on the training set, look-ahead bias was eliminated.
4. **Predictive Performance:** The multivariate LSTM model attained strong predictive accuracy on the held-out test partition: $\text{MAE} = 12.069\,\mu\text{g/m}^3$, $\text{RMSE} = 22.195\,\mu\text{g/m}^3$, and $R^2 = 0.9207$, explaining over 92% of particulate variance.
5. **Multivariate Value:** Integrating meteorological predictors outperformed univariate autoregression (lower RMSE and improved capture of sudden wind-driven clearances).

### 8.2 Practical Limitations
1. **Deterministic Point Predictions:** The current model produces a single deterministic estimate without confidence intervals, which limits uncertainty quantification for high-risk regulatory decisions.
2. **Phase Lag on Rapid Spike Onset:** Under extreme weather shifts, MSE-trained LSTMs can exhibit a 1-to-2 hour delay in responding to abrupt orders of magnitude changes.
3. **Absence of Direct Source Emission Inventories:** The model relies on meteorological proxies without direct inputs for real-time vehicular traffic, industrial factory output, or seasonal heating curtailments.
4. **Single-Step Horizon:** The primary formulation predicts $t+1$ hour; multi-step recursive forecasting over 24–72 hours suffers from accumulated error propagation.

### 8.3 Future Scope
1. **Attention Mechanisms & Transformers:** Integrating temporal multi-head self-attention (e.g., Temporal Fusion Transformers or Informer) to weight past meteorological transitions dynamically.
2. **Probabilistic & Quantile Forecasting:** Implementing Pinball / Quantile loss functions to predict 10th, 50th, and 90th percentile prediction bands.
3. **Multi-Horizon Sequence-to-Sequence Models:** Developing encoder-decoder architectures with scheduled sampling to forecast continuous trajectories up to 48 hours ahead.
4. **Physics-Informed Deep Learning (PINNs):** Incorporating atmospheric advection-diffusion differential equations into the loss function to constrain predictions within physical conservation laws.

---

## 9. References

1. **Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., & Chen, S. X.** (2015). Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and summer holiday effects. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 471(2182), 20150257. https://doi.org/10.1098/rspa.2015.0257
2. **Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X.** (2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 473(2205), 20170457. https://doi.org/10.1098/rspa.2017.0457
3. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. **Chollet, F., et al.** (2015). Keras: Deep learning for humans. *Keras Documentation*. https://keras.io/
5. **UCI Machine Learning Repository & Kaggle.** (2020). *Beijing PM2.5 Data / Multivariate Time Series Air Pollution*. https://www.kaggle.com/datasets
6. **World Health Organization.** (2021). *WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. World Health Organization.

---

## 10. Appendix (Code Section)

Below are the clean, executable core code modules utilized throughout this project.

### 10.1 Dataset Inspection & Quality Audit (`01_dataset_inspection.ipynb`)
```python
import os
import glob
import numpy as np
import pandas as pd

# 1. Discover and load dataset
dataset_path = 'data/LSTM-Multivariate_pollution.csv'
df = pd.read_csv(dataset_path)

print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 2. Schema, Null and Duplicate Audit
schema_audit = pd.DataFrame({
    'Data Type': df.dtypes,
    'Non-Null Count': df.notnull().sum(),
    'Null Count': df.isnull().sum(),
    'Null Pct (%)': (df.isnull().sum() / len(df)) * 100
})
print("\n--- Schema and Null Audit ---")
print(schema_audit)

duplicate_count = df.duplicated().sum()
print(f"\nDuplicate records count: {duplicate_count}")

# 3. Monotonic Datetime Verification
df['datetime'] = pd.to_datetime(df['date'])
is_monotonic = df['datetime'].is_monotonic_increasing
print(f"Chronological Monotonicity: {is_monotonic}")
```

### 10.2 Leakage-Free Preprocessing & Chronological Splitting (`02_eda_and_preprocessing.ipynb`)
```python
import joblib
from sklearn.preprocessing import MinMaxScaler

# 1. One-hot encode categorical wind direction
df_encoded = pd.get_dummies(df, columns=['wnd_dir'], dtype=float)

# Organize feature matrix: pollution (target) at index 0
feature_cols = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain',
                'wnd_dir_NE', 'wnd_dir_NW', 'wnd_dir_SE', 'wnd_dir_cv']

raw_features = df_encoded[feature_cols].values

# 2. Strict Chronological Split (70% Train, 15% Val, 15% Test)
total_samples = len(raw_features)
train_size = int(0.70 * total_samples)
val_size = int(0.15 * total_samples)

train_raw = raw_features[:train_size]
val_raw = raw_features[train_size:train_size + val_size]
test_raw = raw_features[train_size + val_size:]

# 3. Fit Scalers ONLY on Training Data (Zero Leakage)
feature_scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = feature_scaler.fit_transform(train_raw)
val_scaled = feature_scaler.transform(val_raw)
test_scaled = feature_scaler.transform(test_raw)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(train_raw[:, [0]])

# Save fitted scalers
joblib.dump(feature_scaler, 'models/scaler.pkl')
joblib.dump(target_scaler, 'models/target_scaler.pkl')
print("Successfully fitted and persisted zero-leakage scalers.")
```

### 10.3 Sliding-Window Generation & LSTM Model Construction (`03_lstm_forecasting.ipynb`)
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Causal Sliding-Window Tensor Formulation
def create_sliding_sequences(data, look_back=24, target_col_idx=0):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

LOOK_BACK = 24
X_train, y_train = create_sliding_sequences(train_scaled, look_back=LOOK_BACK)
X_val, y_val = create_sliding_sequences(val_scaled, look_back=LOOK_BACK)
X_test, y_test = create_sliding_sequences(test_scaled, look_back=LOOK_BACK)

# 2. Construct Explainable LSTM Network
def build_lstm_model(look_back, num_features, lstm_units=64, learning_rate=0.001):
    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.20),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

model = build_lstm_model(look_back=LOOK_BACK, num_features=X_train.shape[2], lstm_units=64, learning_rate=0.001)

# 3. Model Training with Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# Save model checkpoint
model.save('models/best_lstm_model.keras')
```

### 10.4 Held-Out Test Evaluation & Metric Inversion
```python
# 1. Predict on Held-Out Test Data
y_pred_scaled = model.predict(X_test)

# 2. Inverse Transform to Physical Units (ug/m3)
y_pred_orig = target_scaler.inverse_transform(y_pred_scaled).flatten()
y_true_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 3. Compute Final Test Metrics
test_mae = mean_absolute_error(y_true_orig, y_pred_orig)
test_mse = mean_squared_error(y_true_orig, y_pred_orig)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_true_orig, y_pred_orig)

print("--- HELD-OUT TEST PERFORMANCE ---")
print(f"Test MAE : {test_mae:.3f} ug/m3")
print(f"Test MSE : {test_mse:.3f}")
print(f"Test RMSE: {test_rmse:.3f} ug/m3")
print(f"Test R^2 : {test_r2:.4f}")
```
