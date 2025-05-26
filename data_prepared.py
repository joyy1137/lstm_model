import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
        df_raw = df_raw.drop(['high', 'close','return','K_9_3','D_9_3','J_9_3','MACD_h','RSI',
                              'ma_90','volume_sum','volume_difference','FinanceDifference',
                              'momentum_60'], axis=1)
        df_raw['valuation_date'] = pd.to_datetime(df_raw['valuation_date'])
        df_raw = df_raw.set_index('valuation_date')
        
        # 对原始特征进行标准化
        scaler = MinMaxScaler()
        raw_scaled = scaler.fit_transform(df_raw)
        raw_scaled_df = pd.DataFrame(
            raw_scaled,
            index=df_raw.index,
            columns=df_raw.columns
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
        
        missing_real = [f for f in real_features if f not in raw_scaled_df.columns]
        if missing_real:
            raise ValueError(f"Missing real features: {missing_real}")
        
        # 只选择配置文件中定义的特征
        df_combined = df_combined[discrete_features]
        raw_scaled_df = raw_scaled_df[real_features]
        
        # 合并原始特征和离散特征
        df = pd.merge(df_combined, raw_scaled_df, left_index=True, right_index=True, how='inner')
        
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
        
        # 创建序列数据
        sequence_length = Config.data['sequence_length']
        
        # 创建特征序列
        X_sequences, sequence_dates = DataPrepared.create_sequences(X, sequence_length)
        
        # 获取对应的目标变量
        y_sequences = y.loc[sequence_dates]
        
        return X_sequences, y_sequences, None, sequence_dates

if __name__ == "__main__":
    dp=DataPrepared()
