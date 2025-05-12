import pandas as pd
import numpy as np
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from config import Config
import os

class FeatureEngineering:
    """Handles all feature engineering operations"""
    
    @staticmethod
    def add_technical_indicators(df):
        """Calculate and add technical indicators to the dataframe"""
        df["netval_hs300"] = (1 + df["hs300"]).cumprod()
        df["netval_zz2000"] = (1 + df["zz2000"]).cumprod()
        
        # MACD indicators
        macd_hs300 = ta.macd(df["netval_hs300"].shift(1))
        macd_hs300.rename(columns={'MACDh_12_26_9': 'MACD_diff'}, inplace=True)
        df = pd.concat([df, macd_hs300], axis=1)
        df["macd_signal"] = np.where(df["MACD_diff"] > 0, 1, 0)

        # Bollinger Bands indicators
        bb_zz = ta.bbands(df["netval_zz2000"].shift(1), length=20)
        bb_hs = ta.bbands(df["netval_hs300"].shift(1), length=20)
        boll_percent_zz2000 = (df["netval_zz2000"].shift(1) - bb_zz["BBL_20_2.0"]) / (bb_zz["BBU_20_2.0"] - bb_zz["BBL_20_2.0"])
        boll_percent_hs300 = (df["netval_hs300"].shift(1) - bb_hs["BBL_20_2.0"]) / (bb_hs["BBU_20_2.0"] - bb_hs["BBL_20_2.0"])
        df["boll_percent_diff"] = boll_percent_zz2000 - boll_percent_hs300
       
        # Rate of Change indicators
        roc_zz2000 = ta.roc(df["netval_zz2000"].shift(1), length=10)
        roc_hs300 = ta.roc(df["netval_hs300"].shift(1), length=10)
        df['roc_diff'] = roc_zz2000 - roc_hs300
    
        # Index difference indicators
        df['sz50_gz2000_diff'] = df['gz2000'].shift(1) - df['sz50'].shift(1)
       
        # Clean up intermediate columns
        df.drop(["netval_hs300", "netval_zz2000", "MACD_12_26_9",
                 "MACDs_12_26_9", "MACD_diff"], axis=1, inplace=True)
        
        return df

    @staticmethod
    def prepare_labels(df):
        """Prepare labels for model training"""
        df['zz2000_fwd'] = df['zz2000'].rolling(3).mean().shift(-2)
        df['hs300_fwd'] = df['hs300'].rolling(3).mean().shift(-2)
        df['label'] = df['zz2000_fwd'] - df['hs300_fwd']
        
        return df

    @staticmethod
    def handle_data_leakage(df):
        """Handle data leakage by shifting future-looking features"""
        leak_cols = ["zz1000", "sz50", 'zzA500', 'zz500', 'gz2000']
        for col in leak_cols:
            df[col] = df[col].shift(1)
        # Only drop rows where all leak_cols are NaN
        df.dropna(subset=leak_cols, how='all', inplace=True)
        return df

    @staticmethod
    def generate_majority(df, column, window):
        """Generate majority signal for a given column"""
        diff = df[column]
        signal = (diff > 0).astype(int)
        majority = signal.rolling(window=window).sum()
        df[f"{column}_majority"] = (majority > (window // 2)).astype(int)
        return df

    @staticmethod
    def analyze_feature_correlations(df, real_features, discrete_features, timestamp):
        """Analyze feature correlations using correlation matrix"""
        # Create plots directory if it doesn't exist
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        
        # Combine all features
        all_features = real_features + discrete_features
        feature_data = df[all_features].copy()
        
        # Create correlation matrix
        plt.figure(figsize=(15, 15))
        correlation_matrix = feature_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, f'correlation_matrix_{timestamp}.png'))
        plt.close()
        
        # # Perform PCA - Commented out
        # scaler = MinMaxScaler()
        # scaled_data = scaler.fit_transform(feature_data)
        
        # pca = PCA()
        # pca_result = pca.fit_transform(scaled_data)
        
        # # Plot explained variance ratio
        # plt.figure(figsize=(10, 6))
        # explained_variance_ratio = pca.explained_variance_ratio_
        # cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        # plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
        # plt.xlabel('Number of Components')
        # plt.ylabel('Cumulative Explained Variance Ratio')
        # plt.title('PCA Explained Variance Ratio')
        # plt.grid(True)
        # plt.savefig(f'plots/pca_explained_variance_{timestamp}.png')
        # plt.close()
        
        return None, correlation_matrix

    @staticmethod
    def analyze_high_correlation(corr_matrix, threshold=0.9):
        """Analyze and suggest handling of highly correlated features"""
        suggestions = []
        visited = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    if col1 not in visited and col2 not in visited:
                        visited.update([col1, col2])
                        suggestions.append({
                            "Feature 1": col1,
                            "Feature 2": col2,
                            "Correlation": round(corr_val, 3),
                            "Keep Feature": col1,
                            "Drop Feature": col2,
                            "Suggested Merge Name": f"{col1}_{col2}_mean"
                        })

        return pd.DataFrame(suggestions) 