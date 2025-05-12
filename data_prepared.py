import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import glob
from config import Config

class DataPrepared:
    @staticmethod
    def create_sequences(data, sequence_length):
        """创建时间序列数据"""
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            sequences.append(data.iloc[i:(i + sequence_length)].values)
            targets.append(data.iloc[i + sequence_length].name)  # 使用索引作为目标
        return np.array(sequences), np.array(targets)

    @staticmethod
    def load_and_preprocess_data():
        """加载和预处理数据"""
        # 读取文件
        raw_feature = glob.glob(Config.RAW_FEATURE_PATH)
        df_raw = pd.concat([pd.read_csv(f) for f in raw_feature])
        df_raw['valuation_date'] = pd.to_datetime(df_raw['valuation_date'])
        df_raw = df_raw.set_index('valuation_date')
        
        scaler = MinMaxScaler()
        raw_scaled = scaler.fit_transform(df_raw)
        raw_scaled_df = pd.DataFrame(
            raw_scaled,
            index=df_raw.index,
            columns=df_raw.columns
        )
        
        pca = PCA(n_components=6)
        raw_pca = pca.fit_transform(raw_scaled_df)
        raw_pca_df = pd.DataFrame(
            raw_pca,
            index=df_raw.index,
            columns=[f'pca_{i+1}' for i in range(6)]
        )
        
        feature_files = glob.glob(Config.COMBINE_PATH)
        df_combined = pd.concat([pd.read_csv(f) for f in feature_files])
        df_combined['valuation_date'] = pd.to_datetime(df_combined['valuation_date'])
        df_combined = df_combined.set_index('valuation_date')
        
        discrete_features = Config.DISCRETE_FEATURES
        real_features = Config.REAL_FEATURES
        
        missing_discrete = [f for f in discrete_features if f not in df_combined.columns]
        if missing_discrete:
            raise ValueError(f"Missing discrete features: {missing_discrete}")
        
        missing_real = [f for f in real_features if f not in raw_pca_df.columns]
        if missing_real:
            raise ValueError(f"Missing real features: {missing_real}")
        
        # 只选择配置文件中定义的特征
        df_combined = df_combined[discrete_features]
        raw_pca_df = raw_pca_df[real_features]
        
        # 对 real_features 进行标准化
        real_scaler = MinMaxScaler()
        real_scaled = real_scaler.fit_transform(raw_pca_df)
        raw_pca_df = pd.DataFrame(
            real_scaled,
            index=raw_pca_df.index,
            columns=raw_pca_df.columns
        )
        raw_pca_df=raw_pca_df.shift(1)
        raw_pca_df.dropna(inplace=True)
        
        # 合并 PCA 后的 raw_feature 和 feature_files
        df = pd.merge(df_combined, raw_pca_df, left_index=True, right_index=True, how='inner')
        
        # 读取目标变量数据
        y = pd.read_csv(Config.INDEX_RETURN_PATH, encoding='gbk')
        y.columns = ['valuation_date', 'sz50', 'hs300', 'zz500', 'zz1000', 'zz2000', 'zzA500', 'gz2000']
        y['valuation_date'] = pd.to_datetime(y['valuation_date'])
        y = y.set_index('valuation_date')
        
        # 合并特征和目标变量
        merged = pd.merge(df, y, left_index=True, right_index=True, how='inner')
        merged.sort_index(inplace=True)
        
        # 计算中证2000和沪深300的差值
        merged['zz_hs_diff'] = merged['zz2000'] - merged['hs300']
        merged['zz_hs_diff']=merged['zz_hs_diff'].rolling(3).sum()
        merged['zz_hs_diff']=merged['zz_hs_diff'].shift(-2)
        merged=merged.dropna()
        
        # 创建目标变量：差值大于0为1（选择中证2000），小于0为0（选择沪深300）
        y = (merged['zz_hs_diff'] > 0).astype(int)
        
        # 分离特征和标签
        X = merged.drop(['zz_hs_diff', 'zz2000', 'hs300', 'sz50', 'zz500', 'zz1000', 'zzA500', 'gz2000'], axis=1)
        
        # 分离连续特征和离散特征
        continuous_features = X.select_dtypes(include=['float64', 'int64']).columns
        discrete_features = X.select_dtypes(include=['object', 'category']).columns
        
        # 处理离散特征
        if len(discrete_features) > 0:
            # 对离散特征进行编码
            X_discrete = pd.get_dummies(X[discrete_features])
            X_continuous = X[continuous_features]
            X = pd.concat([X_continuous, X_discrete], axis=1)
        else:
            X_discrete = None
        
        # 创建序列数据
        sequence_length = Config.data['sequence_length']
        
        # 首先创建特征序列
        X_sequences, sequence_dates = DataPrepared.create_sequences(X, sequence_length)
        
        # 然后获取对应的目标变量
        y_sequences = y.loc[sequence_dates]
        
        return X_sequences, y_sequences, X_discrete, sequence_dates

if __name__ == "__main__":
    dp=DataPrepared()
