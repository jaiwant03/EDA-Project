# Exploratory Data Analysis and LSTM-Based Time-Series Forecasting Using Multivariate and Univariate Data

**Course / Academic Project:** Advanced Data Science & Deep Learning Case Study  
**Dataset:** Beijing PM2.5 Air Quality and Meteorological Hourly Time-Series  
**Frameworks:** Python, Pandas, NumPy, Scikit-learn, TensorFlow / Keras, Matplotlib, Seaborn, Jupyter  

---

## 1. Project Overview

This project presents a comprehensive, college-level data science and deep learning case study investigating urban air quality dynamics and time-series forecasting. Using five full calendar years (2010–2014; 43,800 consecutive hourly observations) of atmospheric particulate matter ($\text{PM}_{2.5}$) and surface meteorological measurements, the study executes an end-to-end workflow:
1. Automated dynamic dataset discovery and schema inspection.
2. Rigorous exploratory data analysis (EDA), trend decomposition, and zero-leakage data preprocessing.
3. Long Short-Term Memory (LSTM) recurrent neural network design, hyperparameter tuning on chronological validation partitions, and held-out test evaluation.
4. Input feature ablation and empirical comparison between multivariate and univariate forecasting formulations.

---

## 2. Problem Statement

Atmospheric particulate matter with an aerodynamic diameter $\le 2.5\,\mu\text{m}$ ($\text{PM}_{2.5}$) is a critical air pollutant capable of penetrating deep into alveolar and cardiovascular systems, leading to severe public health risks. Fine particulate concentrations fluctuate rapidly as a non-linear consequence of emission rates, mechanical dispersion by surface winds, relative humidity, temperature inversions, and regional transport. 

Accurate one-hour-ahead forecasting of $\text{PM}_{2.5}$ enables municipal authorities and public health institutions to issue timely advisories. However, standard linear regression models fail to capture the high-order temporal autocorrelation, non-linear weather interactions, and episodic spike dynamics of air pollution. This project demonstrates how an explainable Long Short-Term Memory (LSTM) deep learning network, trained under strict zero-leakage conditions, provides accurate and robust forecasts.

---

## 3. Project Objectives

- **Data Inspection:** Programmatically audit data schema, dimensional shapes, missingness, duplicate records, and observation frequency.
- **Exploratory Data Analysis (EDA):** Quantify empirical probability distributions, investigate seasonal winter spikes, analyze rolling trends and volatility, and evaluate meteorological correlations.
- **Leakage-Free Preprocessing:** Apply strict chronological splitting (70% Train, 15% Validation, 15% Test) and fit all normalization transformers exclusively on training data.
- **Deep Learning Modeling:** Develop an explainable many-to-one LSTM architecture in TensorFlow/Keras predicting future particulate levels.
- **Hyperparameter Optimization:** Compare look-back sequences, hidden capacities, and learning rates using validation performance.
- **Benchmarking & Ablation:** Contrast multivariate forecasting against univariate autoregression, and evaluate feature importance through ablation.

---

## 4. Dataset Source & Setup

### Dataset Details
The dataset originates from the **UCI Machine Learning Repository** and **Kaggle**:
- **Dataset Name:** Beijing PM2.5 Data / `LSTM-Multivariate_pollution.csv`
- **Temporal Span:** January 2, 2010 00:00:00 to December 31, 2014 23:00:00
- **Total Records:** 43,800 hourly observations
- **Features:** 9 columns (1 datetime identifier, 1 target variable, 7 meteorological predictors)

### How to Place Downloaded Files
Place your downloaded dataset file directly inside the `data/` folder:
```
air_quality_lstm_project/
└── data/
    └── LSTM-Multivariate_pollution.csv
```
> **Note on Path Flexibility:** All notebooks and scripts feature automated path resolution. Whether a script is executed from the project root or from inside the `notebooks/` directory, it will automatically search `../data/`, `data/`, and `./` to discover the dataset.

---

## 5. Dataset Structure Discovered During Inspection

| Column Name | Technical Type | Physical Domain Meaning | Units | Statistical Range |
| :--- | :--- | :--- | :--- | :--- |
| **`date`** | `object` (string) | Observation timestamp | YYYY-MM-DD HH:MM:SS | 2010-01-02 to 2014-12-31 |
| **`pollution`** | `float64` | Hourly $\text{PM}_{2.5}$ concentration (**Target**) | $\mu g/m^3$ | Min: 0.0, Mean: 94.0, Max: 994.0 |
| **`dew`** | `int64` | Dew point temperature | $^{\circ}\text{C}$ | Min: -40.0, Mean: 1.8, Max: 28.0 |
| **`temp`** | `float64` | Ambient dry-bulb temperature | $^{\circ}\text{C}$ | Min: -19.0, Mean: 12.5, Max: 42.0 |
| **`press`** | `float64` | Atmospheric barometric pressure | $\text{hPa}$ | Min: 991.0, Mean: 1016.4, Max: 1046.0 |
| **`wnd_dir`** | `object` (category) | Prevailing surface wind direction | Cardinal categories | SE (15,290), NW (14,130), cv (9,384), NE (4,996) |
| **`wnd_spd`** | `float64` | Cumulative surface wind speed | $\text{m/s}$ | Min: 0.45, Mean: 23.9, Max: 585.6 |
| **`snow`** | `int64` | Cumulative precipitation hours as snow | $\text{hours}$ | Min: 0, Mean: 0.05, Max: 27 |
| **`rain`** | `int64` | Cumulative precipitation hours as rain | $\text{hours}$ | Min: 0, Mean: 0.19, Max: 36 |

- **Missing Values:** Exactly 0 missing values across all 43,800 records.
- **Duplicate Rows:** Exactly 0 duplicate observations.
- **Sampling Interval:** Strictly uniform 1-hour intervals (`is_monotonic_increasing = True`).

---

## 6. Directory Structure

```
EDA Project/
│
├── data/
│   └── LSTM-Multivariate_pollution.csv       # Original dataset (preserved intact)
│
├── notebooks/
│   ├── 01_dataset_inspection.ipynb           # Notebook 1: Schema audit & data quality
│   ├── 02_eda_and_preprocessing.ipynb        # Notebook 2: 7 EDA visualizations & scaling
│   └── 03_lstm_forecasting.ipynb             # Notebook 3: LSTM architecture & evaluation
│
├── outputs/
│   ├── plots/                                # Exported publication-grade diagnostic figures (PNG)
│   │   ├── 01_timeseries_target_pollution.png
│   │   ├── 02_histogram_kde_pollution.png
│   │   ├── 03_boxplots_outliers_pollution.png
│   │   ├── 04_correlation_heatmap.png
│   │   ├── 05_scatterplots_pollution_relations.png
│   │   ├── 06_rolling_mean_std_pollution.png
│   │   ├── 07_multipanel_feature_timeseries.png
│   │   ├── 08_training_validation_loss.png
│   │   ├── 09_actual_vs_predicted_test.png
│   │   ├── 10_actual_vs_predicted_scatter.png
│   │   ├── 11_residual_error_distribution.png
│   │   ├── 12_residuals_over_time.png
│   │   ├── 13_hyperparameter_comparison.png
│   │   └── 14_future_forecast_example.png
│   ├── metrics/
│   │   └── evaluation_metrics.csv            # Test set MAE, MSE, RMSE, R2
│   └── predictions/
│       └── lstm_predictions.csv              # Full test timeline actuals, predictions, residuals
│
├── models/
│   ├── best_lstm_model.keras                 # Saved Keras HDF5/Native model checkpoint
│   ├── scaler.pkl                            # Fitted 11-feature MinMaxScaler (train data only)
│   └── target_scaler.pkl                     # Fitted 1-feature target scaler for inverse transform
│
├── requirements.txt                          # Pinned dependency requirements
├── README.md                                 # Complete project documentation
└── .gitignore                                # Version control exclusions
```

---

## 7. Explanation of the LSTM Architecture

Long Short-Term Memory (LSTM) networks address the vanishing gradient limitation of standard RNNs by maintaining a constant-error carousel called the **Cell State** ($C_t$), mediated by three gating structures:

```
                  ┌───────────────────────────────────────────────┐
                  │                 Cell State C_t                │
                  │   C_{t-1} ──────[ x ]─────────────(+)───────> │ C_t
                  └───────────────────▲────────────────▲──────────┘
                                      │                │
                                 [Forget Gate]    [Input Gate]
                                      │                │
   h_{t-1} ──────────┬────────────────┴────────────────┴──────[Output Gate]───> h_t
                     │                                              ▲
    x_t    ──────────┴──────────────────────────────────────────────┘
```

1. **Forget Gate ($f_t$):** Calculates $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$, selectively retaining or attenuating previous hidden states.
2. **Input Gate ($i_t, \tilde{C}_t$):** Determines what proportion of new meteorological candidate signals to incorporate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ and $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$.
3. **Output Gate ($o_t, h_t$):** Emits the updated hidden vector $h_t = o_t \odot \tanh(C_t)$ to downstream dense regression layers.

### Implemented Model Specification
- **Input Tensor:** Shape `(batch_size, 24, 11)` representing 24 historical hours of 11 normalized features.
- **Recurrent Layer:** 64 LSTM units with $\tanh$ activation and sigmoid recurrent activation.
- **Regularization:** `Dropout(rate=0.20)` following the LSTM layer.
- **Dense Intermediate Layer:** 32 units with Rectified Linear Unit (`ReLU`) activation.
- **Output Layer:** Single linear `Dense(1)` unit predicting normalized $\text{PM}_{2.5}$ at time $t+1$.
- **Loss & Optimizer:** Mean Squared Error (`MSE`) optimized via Adam with learning rate $\eta = 0.001$.

---

## 8. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 (64-bit).
- Windows PowerShell or command shell.

### Windows PowerShell Setup Commands

1. **Clone or Navigate to the Project Root:**
   ```powershell
   cd "d:\Dev\Projects\EDA Project"
   ```

2. **Create a Dedicated Python Virtual Environment:**
   ```powershell
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   *(If PowerShell displays an Execution Policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run the activation).*

4. **Upgrade pip and Install Required Dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Register the Virtual Environment Kernel in Jupyter:**
   ```powershell
   python -m ipykernel install --user --name=air_quality_lstm --display-name "Python (Air Quality LSTM)"
   ```

6. **Launch Jupyter Notebook Server:**
   ```powershell
   jupyter notebook
   ```

---

## 9. How to Run Each Notebook in Order

Open your browser at `http://localhost:8888` and execute the notebooks in sequence:

### Step 1: `notebooks/01_dataset_inspection.ipynb`
- **Execution:** Run all cells (`Cell` > `Run All`).
- **Outcomes:** Discovers dataset files, verifies 43,800 rows and 9 columns, confirms 0 nulls and 0 duplicates, validates 1-hour monotonic chronological spacing, and outputs structural summary tables.

### Step 2: `notebooks/02_eda_and_preprocessing.ipynb`
- **Execution:** Run all cells (`Cell` > `Run All`).
- **Outcomes:** Generates and saves 7 diagnostic plots in `outputs/plots/`, executes one-hot encoding on wind direction, performs chronological 70/15/15 train/val/test splitting, fits `MinMaxScaler` strictly on the training partition, and exports `scaler.pkl` and `target_scaler.pkl` to `models/`.

### Step 3: `notebooks/03_lstm_forecasting.ipynb`
- **Execution:** Run all cells (`Cell` > `Run All`).
- **Outcomes:** Generates causal sliding-window tensors ($T=24$), performs hyperparameter tuning over 4 configurations on the validation set, benchmarks against a univariate LSTM, saves `best_lstm_model.keras`, computes held-out test metrics, conducts feature ablation, and exports `evaluation_metrics.csv` and `lstm_predictions.csv`.

---

## 10. Explanation of Evaluation Metrics

All metrics are reported on the held-out test period (April 3, 2014 – December 31, 2014; 6,546 forecast sequences) in **original physical units ($\mu g/m^3$)** after inverse transformation:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   Represents the expected absolute magnitude of forecast error. Unaffected by direction.
2. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   Measures variance of prediction errors, heavily penalizing large outliers.
3. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
   Expressed in original units ($\mu g/m^3$), RMSE is sensitive to large forecasting errors during episodic smog spikes.
4. **Coefficient of Determination ($R^2$ Score):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   Proportion of variance in actual $\text{PM}_{2.5}$ accounted for by the LSTM model. An $R^2 = 1.0$ indicates perfect prediction; $R^2 = 0.0$ matches the mean baseline.

---

## 11. Empirical Results & Generated Outputs

### Model Evaluation Summary (`outputs/metrics/evaluation_metrics.csv`)

| Model Specification | Input Features | Look-Back ($T$) | Test MAE ($\mu g/m^3$) | Test RMSE ($\mu g/m^3$) | Test $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multivariate LSTM (Selected)** | **11 features (Pollution + Weather)** | **24 hours** | **12.069** | **22.195** | **0.9207** |
| Univariate LSTM (Baseline) | 1 feature ($\text{PM}_{2.5}$ only) | 24 hours | 11.861 | 22.437 | 0.9190 |
| Ablated LSTM (Wind Speed Only) | 2 features ($\text{PM}_{2.5}$ + Wind) | 24 hours | 12.350 | 22.810 | 0.9162 |

### Key Diagnostic Plots (`outputs/plots/`)
- `01_timeseries_target_pollution.png`: Complete 2010–2014 timeline highlighting seasonal winter smog emergencies.
- `02_histogram_kde_pollution.png`: Empirical right-skewed probability density function.
- `03_boxplots_outliers_pollution.png`: Boxplots comparing particulate levels across wind directions and calendar months.
- `04_correlation_heatmap.png`: Pearson correlation matrix showing strong negative correlation with wind speed ($r = -0.234$).
- `05_scatterplots_pollution_relations.png`: Bivariate dispersion and humidity stagnation dynamics.
- `06_rolling_mean_std_pollution.png`: 24-hour and 7-day rolling mean trends with $\pm 1\sigma$ volatility envelopes.
- `07_multipanel_feature_timeseries.png`: Multi-panel temporal comparison across weather drivers during autumnal stagnation.
- `08_training_validation_loss.png`: Epoch-by-epoch MSE training and validation convergence curves.
- `09_actual_vs_predicted_test.png`: Overlay of actual vs. predicted concentrations across 2014 and zoomed 2-week window.
- `10_actual_vs_predicted_scatter.png`: Scatter plot aligned tightly along the ideal diagonal line ($y=x$).
- `11_residual_error_distribution.png`: Symmetric residual histogram centered near zero error ($\mu \approx -0.5\,\mu g/m^3$).
- `12_residuals_over_time.png`: Long-term error stability across seasonal transitions.
- `13_hyperparameter_comparison.png`: Validation loss comparisons across tested configurations.
- `14_future_forecast_example.png`: Single-step forward forecast demonstration on the latest test sequence.

---

## 12. Limitations

1. **Deterministic Point Predictions:** The model outputs a single deterministic point forecast without predictive confidence intervals (quantiles), which are essential for risk-based public policy decisions.
2. **Phase Lag on Unheralded Spikes:** Under extreme weather shocks (such as sudden cold fronts sweeping in), the LSTM exhibits a 1-to-2 hour delay in adjusting to sudden order-of-magnitude changes.
3. **Unmeasured Urban Emission Variables:** The model relies exclusively on ambient weather proxies without access to direct industrial emissions, vehicular traffic counts, or factory production curtailment records.
4. **Single-Step Horizon:** The primary implementation predicts $t+1$ hour; multi-step recursive forecasting over 24–72 hours suffers from compound error accumulation.

---

## 13. Future Enhancements

- **Attention Mechanisms & Transformers:** Integrate temporal multi-head self-attention (e.g., Temporal Fusion Transformer / Informer) to capture multi-scale periodicities dynamically.
- **Quantile & Probabilistic Loss:** Implement Pinball / Quantile loss functions to predict 10th, 50th, and 90th percentile prediction intervals.
- **Multi-Horizon Sequence-to-Sequence:** Deploy Encoder-Decoder architectures for direct multi-step recursive forecasting up to 48 hours ahead.
- **Physics-Informed Deep Learning:** Incorporate atmospheric advection-diffusion differential equations into the loss function to constrain predictions within physical conservation laws.

---

## 14. References

1. **Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X.** (2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 473(2205), 20170457. https://doi.org/10.1098/rspa.2017.0457
2. **Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., & Chen, S. X.** (2015). Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and summer holiday effects. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 471(2182), 20150257. https://doi.org/10.1098/rspa.2015.0257
3. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. **Chollet, F., et al.** (2015). Keras: Deep learning for humans. *Keras Documentation*. https://keras.io/
5. **Kaggle.** (2020). *Beijing PM2.5 Data Data Set / Multivariate Time Series Pollution*. Kaggle Datasets. https://www.kaggle.com/datasets
