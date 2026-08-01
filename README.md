# Anomaly Detection in IoT Sensors - Machine Learning Project

## Project Overview
This project implements multiple anomaly detection methods to identify anomalies in IoT sensor data using real-world data from Kaggle's Pump Sensor dataset.

## Objectives
- Detect anomalies in sensor readings
- Compare multiple anomaly detection techniques
- Identify unusual patterns in IoT data

## Dataset
- **Source**: Kaggle - Pump Sensor Data
- **Size**: 220,320 rows, 55 columns
- **Features**: 52 sensor readings (sensor_00 to sensor_51), machine_status
- **Type**: Time-series sensor data

## Methodology

### 1. Data Loading & Preprocessing
- Loaded Kaggle pump sensor dataset
- Selected relevant numerical features
- Handled missing values (median imputation)
- Standardized features (StandardScaler)

### 2. Exploratory Data Analysis (EDA)
- Sensor reading distributions
- Correlation analysis between sensors
- Time-series pattern visualization
- Outlier identification

### 3. Anomaly Detection Methods

| Method | Type | Description |
|--------|------|-------------|
| **Isolation Forest** | Unsupervised | Isolates anomalies using random forests |
| **DBSCAN** | Clustering | Density-based clustering, outliers as noise |
| **Statistical (Z-Score)** | Statistical | Flags data points > 3 std from mean |

### 4. Model Evaluation
- **Accuracy**: Overall detection accuracy
- **Precision**: True anomalies / Detected anomalies
- **Recall**: True anomalies / Actual anomalies
- **F1 Score**: Harmonic mean of precision and recall

## Results (Unsupervised - No Labels)

| Method | Anomalies Detected | Percentage |
|--------|-------------------|------------|
| Isolation Forest | 250 | 5.0% |
| DBSCAN | 5,000 | 100.0% |
| Statistical (Z-Score) | 2,070 | 41.4% |

**Best Method**: Isolation Forest (5% detection rate - typical for anomaly detection)

## Key Findings
1. Isolation Forest provides the most reasonable anomaly rate (~5%)
2. DBSCAN with default parameters labels all points as anomalies (needs tuning)
3. Z-Score method detects 41% as anomalies (threshold may need adjustment)

## Files Generated
- `anomaly_detection_iot.py` - Main analysis script
- `raw_data.csv` - Raw sensor data
- `iot_eda.png` - Exploratory data analysis plots
- `sensor_correlation.png` - Sensor correlation heatmap
- `anomaly_detection_results.png` - Anomaly detection visualization
- `method_comparison.png` - Method performance comparison

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
python anomaly_detection_iot.py
```

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- kagglehub (for data download)

## Real-World Applications
1. **Predictive Maintenance**: Detect equipment failures before they happen
2. **Quality Control**: Identify defective products in manufacturing
3. **Network Security**: Detect intrusions or attacks
4. **Healthcare**: Monitor patient vital signs for anomalies
5. **Financial fraud**: Detect unusual transaction patterns
