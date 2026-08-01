import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)


def load_kaggle_iot_data():
    try:
        import kagglehub
        path = kagglehub.dataset_download("nphantawee/pump-sensor-data")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(path, csv_files[0]))
            print(f"  Loaded from Kaggle: {df.shape[0]} rows")
            return df
    except Exception as e:
        print(f"  Kaggle failed: {e}")
    return None


def load_kaggle_anomaly_data():
    try:
        import kagglehub
        path = kagglehub.dataset_download("disham991/real-time-anomaly-detection-in-iot")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(path, csv_files[0]))
            print(f"  Loaded from Kaggle: {df.shape[0]} rows")
            return df
    except Exception as e:
        print(f"  Kaggle failed: {e}")
    return None


def load_sklearn_data():
    try:
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        return df
    except:
        return None


def generate_iot_data(n_points=5000):
    np.random.seed(42)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='5min')
    
    temperature = 22 + 5 * np.sin(2 * np.pi * np.arange(n_points) / 288) + np.random.normal(0, 0.5, n_points)
    humidity = 55 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 288 + np.pi/4) + np.random.normal(0, 1.0, n_points)
    pressure = 1013 + 2 * np.sin(2 * np.pi * np.arange(n_points) / 1440) + np.random.normal(0, 0.3, n_points)
    vibration = 0.5 + 0.2 * np.sin(2 * np.pi * np.arange(n_points) / 100) + np.random.normal(0, 0.05, n_points)
    power = 150 + 30 * np.sin(2 * np.pi * np.arange(n_points) / 288) + np.random.normal(0, 3, n_points)
    
    n_anomalies = int(n_points * 0.05)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    anomaly_type = np.random.choice(['spike', 'drop', 'drift'], n_anomalies)
    
    for idx, a_type in zip(anomaly_indices, anomaly_type):
        if a_type == 'spike':
            temperature[idx] += np.random.uniform(8, 15)
            humidity[idx] += np.random.uniform(15, 25)
            vibration[idx] += np.random.uniform(1.0, 3.0)
            power[idx] += np.random.uniform(50, 100)
        elif a_type == 'drop':
            temperature[idx] -= np.random.uniform(8, 15)
            humidity[idx] -= np.random.uniform(15, 25)
            power[idx] -= np.random.uniform(50, 100)
    
    is_anomaly = np.zeros(n_points, dtype=int)
    is_anomaly[anomaly_indices] = 1
    
    return pd.DataFrame({
        'Timestamp': timestamps,
        'Temperature': np.round(temperature, 2),
        'Humidity': np.round(humidity, 2),
        'Pressure': np.round(pressure, 2),
        'Vibration': np.round(vibration, 4),
        'Power_Consumption': np.round(power, 2),
        'Is_Anomaly': is_anomaly
    })


def load_real_data():
    print("\n[1] Loading IoT sensor dataset...")
    
    loaders = [
        ("Kaggle Pump Sensor", load_kaggle_iot_data),
        ("Kaggle IoT Anomaly", load_kaggle_anomaly_data),
        ("Synthetic IoT Data", generate_iot_data),
    ]
    
    for name, loader in loaders:
        try:
            print(f"\n  Trying: {name}...")
            df = loader()
            if df is not None and len(df) > 100:
                print(f"  SUCCESS: {name}")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                return df, name
        except Exception as e:
            print(f"  Failed: {e}")
    
    return generate_iot_data(), "Synthetic IoT Data"


def perform_eda(df, feature_cols):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    for idx, sensor in enumerate(feature_cols[:5]):
        row, col = idx // 2, idx % 2
        axes[row, col].plot(df.index[:500], df[sensor][:500], alpha=0.7, linewidth=0.8)
        axes[row, col].set_title(f'{sensor} Readings')
        axes[row, col].set_ylabel(sensor)
    
    if len(feature_cols) >= 6:
        axes[2, 0].hist(df[feature_cols[0]], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[2, 0].set_title(f'{feature_cols[0]} Distribution')
    
    target_col = 'Is_Anomaly' if 'Is_Anomaly' in df.columns else df.columns[-1]
    if target_col in df.columns and df[target_col].nunique() <= 10:
        anomaly_counts = df[target_col].value_counts()
        axes[2, 1].pie(anomaly_counts.values, labels=anomaly_counts.index.astype(str),
                        autopct='%1.1f%%', colors=['steelblue', 'coral'])
        axes[2, 1].set_title('Anomaly Distribution')
    
    plt.tight_layout()
    plt.savefig('iot_eda.png', dpi=150)
    plt.show()


def plot_correlation(df, features):
    plt.figure(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Sensor Correlation Matrix')
    plt.tight_layout()
    plt.savefig('sensor_correlation.png', dpi=150)
    plt.show()


def isolation_forest_detection(X_scaled, contamination=0.05):
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            max_samples='auto', random_state=42, n_jobs=-1)
    predictions = model.fit_predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)
    return predictions, anomaly_scores, model


def dbscan_detection(X_scaled, eps=0.5, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    clusters = model.fit_predict(X_scaled)
    predictions = np.where(clusters == -1, -1, 1)
    return predictions, clusters, model


def statistical_detection(df, features, threshold=3):
    predictions = np.ones(len(df))
    anomaly_scores = np.zeros(len(df))
    
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        if std > 0:
            z_scores = np.abs((df[feature] - mean) / std)
            anomaly_mask = z_scores > threshold
            predictions[anomaly_mask] = -1
            anomaly_scores = np.maximum(anomaly_scores, z_scores.values)
    
    return predictions, anomaly_scores


def evaluate_detection(y_true, y_pred, method_name):
    y_pred_binary = np.where(y_pred == -1, 1, 0)
    
    print(f"\n{'='*50}")
    print(f"Method: {method_name}")
    print(f"{'='*50}")
    
    cm = confusion_matrix(y_true, y_pred_binary)
    print(f"Confusion Matrix:")
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    if cm[1,1] + cm[0,1] > 0 and cm[1,1] + cm[1,0] > 0:
        precision = cm[1,1] / (cm[1,1] + cm[0,1])
        recall = cm[1,1] / (cm[1,1] + cm[1,0])
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
    
    return {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}


def plot_anomaly_detection_results(df, predictions_dict, feature_col='Temperature'):
    n_show = min(500, len(df))
    fig, axes = plt.subplots(len(predictions_dict), 1, figsize=(16, 5*len(predictions_dict)))
    
    if len(predictions_dict) == 1:
        axes = [axes]
    
    for row, (method_name, predictions) in enumerate(predictions_dict.items()):
        axes[row].plot(range(n_show), df[feature_col].values[:n_show], alpha=0.7, linewidth=0.8, label='Normal')
        
        anomaly_mask = predictions[:n_show] == -1
        if anomaly_mask.any():
            axes[row].scatter(np.where(anomaly_mask)[0],
                              df[feature_col].values[:n_show][anomaly_mask],
                              color='red', s=30, zorder=5, label='Detected Anomaly')
        
        axes[row].set_title(f'{method_name} - {feature_col}')
        axes[row].legend()
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=150)
    plt.show()


def plot_comparison(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    results_df[['Precision', 'Recall', 'F1']].plot(kind='bar', ax=axes[0],
                                                      color=['steelblue', 'coral', 'teal'],
                                                      edgecolor='black')
    axes[0].set_title('Detection Performance Comparison')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=0)
    
    results_df['Accuracy'].plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
    axes[1].set_title('Accuracy Comparison')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150)
    plt.show()


def main():
    print("=" * 70)
    print("  ANOMALY DETECTION IN IOT SENSORS - ML PROJECT")
    print("  Real Data + Multiple Methods")
    print("=" * 70)
    
    df, data_source = load_real_data()
    print(f"\n  Dataset: {data_source}")
    
    target_col = 'Is_Anomaly' if 'Is_Anomaly' in df.columns else None
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col and c != 'Timestamp'][:5]
    
    print(f"  Features: {feature_cols}")
    
    df.to_csv('raw_data.csv', index=False)
    
    print("\n[2] Exploratory Data Analysis...")
    perform_eda(df, feature_cols)
    plot_correlation(df, feature_cols[:5])
    
    print("\n[3] Preprocessing data...")
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c != target_col and c != 'Unnamed: 0']
    
    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())
    df_numeric = df_numeric.dropna(axis=1, how='all')
    
    df_clean = df_numeric.head(5000).copy()
    feature_cols_clean = list(df_clean.columns)
    
    print(f"  Clean samples: {len(df_clean)}")
    print(f"  Features: {len(feature_cols_clean)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean.values)
    X_scaled = np.nan_to_num(X_scaled, nan=0)
    
    if target_col and target_col in df.columns:
        y_true = df.loc[df_clean.index, target_col].values if len(df_clean) > 0 else np.zeros(len(df_clean))
    else:
        y_true = np.zeros(len(df_clean))
        print("  Warning: No anomaly labels found, using unsupervised evaluation")
    
    print(f"\n[4] Running Anomaly Detection Methods...")
    
    results = {}
    predictions_dict = {}
    
    print("\n  -> Method 1: Isolation Forest...")
    iso_pred, iso_scores, iso_model = isolation_forest_detection(X_scaled, contamination=0.05)
    if target_col:
        results['Isolation Forest'] = evaluate_detection(y_true, iso_pred, 'Isolation Forest')
    predictions_dict['Isolation Forest'] = iso_pred
    
    print("\n  -> Method 2: DBSCAN Clustering...")
    db_pred, db_clusters, db_model = dbscan_detection(X_scaled, eps=0.8, min_samples=10)
    if target_col:
        results['DBSCAN'] = evaluate_detection(y_true, db_pred, 'DBSCAN')
    predictions_dict['DBSCAN'] = db_pred
    
    print("\n  -> Method 3: Statistical Z-Score...")
    stat_pred, stat_scores = statistical_detection(df_clean, feature_cols_clean, threshold=3)
    if target_col:
        results['Statistical (Z-Score)'] = evaluate_detection(y_true, stat_pred, 'Statistical (Z-Score)')
    predictions_dict['Statistical (Z-Score)'] = stat_pred
    
    print("\n[5] Generating Visualizations...")
    
    if results:
        results_df = pd.DataFrame(results).T
        print("\n" + results_df.to_string())
        plot_comparison(results_df)
        results_df.to_csv('detection_comparison.csv')
        
        best_method = results_df['F1'].idxmax()
        print(f"\n{'='*60}")
        print(f"  BEST METHOD: {best_method} (F1: {results_df.loc[best_method, 'F1']:.4f})")
        print(f"{'='*60}")
    else:
        print("\n  No labeled data for evaluation, showing detection results only")
    
    plot_anomaly_detection_results(df_clean, predictions_dict, feature_cols_clean[0])
    
    print("\n[6] Anomaly Statistics...")
    for method, preds in predictions_dict.items():
        n_anomalies = (preds == -1).sum()
        print(f"  {method}: {n_anomalies} anomalies detected ({n_anomalies/len(preds)*100:.1f}%)")
    
    print("\n  PROJECT COMPLETE!")


if __name__ == "__main__":
    main()
