import pandas as pd
import numpy as np
from config import Config
import os
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml
from datetime import datetime
from neural_network_trainer import NeuralNetworkTrainer
from signal_backtesting.backtester import run_backtesting
import torch
from data_prepared import DataPrepared
from sklearn.model_selection import TimeSeriesSplit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.FEATURE_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    with open('config/parameters.yml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    X_sequences, y_sequences, X_discrete, sequence_dates = DataPrepared.load_and_preprocess_data()
    dates = pd.Series(pd.to_datetime(sequence_dates))
    y_sequences = y_sequences.values
    
    
    
    # 用于存储所有预测结果
    all_predictions = []
    all_test_dates = []
    all_test_values = []
    
    trainer = NeuralNetworkTrainer(random_state=Config.random_state)
    
    # 对每个日期范围训练模型
    for i, (date_key, date_range) in enumerate(params['dates'].items()):
        print(f"\nTraining model for {date_key}")
        print(f"Train period: {date_range['train_start']} to {date_range['train_end']}")
        print(f"Test period: {date_range['test_start']} to {date_range['test_end']}")
        
        train_start = pd.to_datetime(date_range['train_start'])
        train_end = pd.to_datetime(date_range['train_end'])
        test_start = pd.to_datetime(date_range['test_start'])
        test_end = pd.to_datetime(date_range['test_end'])
        
        print(f"\nDate ranges:")
        print(f"Train period: {train_start} to {train_end}")
        print(f"Test period: {test_start} to {test_end}")
        
        # 根据日期范围划分训练集和测试集
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        
        # 检查训练集是否为空
        if sum(train_mask) == 0:
            print(f"\nWarning: Train set is empty for {date_key}. Skipping this period.")
            continue
            
        print(f"\nMatching dates:")
        print(f"Training dates: {dates[train_mask].iloc[0]} to {dates[train_mask].iloc[-1]}")
        if sum(test_mask) > 0:
            print(f"Test dates: {dates[test_mask].iloc[0]} to {dates[test_mask].iloc[-1]}")
        print(f"Number of training dates: {sum(train_mask)}")
        print(f"Number of test dates: {sum(test_mask)}")
        
        # 确保数据长度匹配
        if len(X_sequences) != len(dates):
            print(f"\nWarning: Data length ({len(X_sequences)}) doesn't match date range length ({len(dates)})")
            # 如果数据长度不匹配，使用较短的序列
            min_length = min(len(X_sequences), len(dates))
            X_sequences = X_sequences[:min_length]
            y_sequences = y_sequences[:min_length]
            dates = dates[:min_length]
            train_mask = train_mask[:min_length]
            test_mask = test_mask[:min_length]
        
        
        # 划分训练集和测试集
        X_train_full = X_sequences[train_mask]
        y_train_full = y_sequences[train_mask]
        X_test = X_sequences[test_mask]
        y_test = y_sequences[test_mask]
        
        
        
        # 使用TimeSeriesSplit进行训练集和验证集的划分
        tscv = TimeSeriesSplit(n_splits=Config.n_splits)
        for train_idx, val_idx in tscv.split(X_train_full):
            X_train = X_train_full[train_idx]
            y_train = y_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]
            break  # 只使用第一个划分，因为我们在循环中训练模型
        
    
        
        # 训练模型
        print(f"\nTraining model for {date_key}...")
        model = trainer.train_model(
            X_train, y_train,
            X_val, y_val,  # 使用验证集而不是测试集
            discrete_features_train=None,
            discrete_features_val=None,
            use_grid_search=True,
            config_name=f"model_{date_key}"
        )
        
        # 保存模型
        model_path = os.path.join(Config.MODELS_DIR, f'model_{date_key}.path')
        trainer.save_model(model, model_path)
        
        # 进行预测
        model.eval()
        with torch.no_grad():
            # 使用训练好的模型进行预测
            model, _ = trainer.get_trained_model_and_scaler()
            
            # 检查测试集是否为空
            if len(X_test) == 0:
                continue
            
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_pred = model(X_test_tensor).cpu().numpy()
            y_test_pred = y_test_pred.reshape(-1)  # 确保是1维数组
            y_test_pred = (y_test_pred > 0.5).astype(int)  # 将概率转换为0/1预测
            
            # 收集预测结果
            all_predictions.extend(y_test_pred)
            all_test_dates.extend(dates[test_mask].values)  
            all_test_values.extend(y_test)
    
    # 绘制所有模型的准确率和损失曲线
    trainer.plot_all_models_curves()
    
    # 创建完整的预测结果DataFrame
    predictions_df = pd.DataFrame({
        'valuation_date': all_test_dates,
        'prediction': all_predictions
    })
    predictions_df = predictions_df.sort_values('valuation_date')
    
    # 检查是否有预测结果
    if len(predictions_df) == 0:
        print("\nNo predictions available for backtesting. Skipping backtesting step.")
        return
    
    # 保存预测结果
    os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
    predictions_df.to_csv(Config.TEST_PRED_PATH, index=False)
    
    # 获取完整的收益率数据
    returns = pd.read_csv(Config.INDEX_RETURN_PATH, encoding='gbk')
    returns['valuation_date'] = pd.to_datetime(returns['valuation_date'])
    returns = returns.set_index('valuation_date')
    
    # 确保预测结果和收益率数据使用相同的日期
    common_dates = pd.Index(predictions_df['valuation_date']).intersection(returns.index)
    predictions_df = predictions_df[predictions_df['valuation_date'].isin(common_dates)]
    returns = returns.loc[common_dates]
    
    # 检查是否有共同日期
    if len(common_dates) == 0:
        print("\nNo common dates between predictions and returns data. Skipping backtesting step.")
        return
    
    # 运行回测
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_backtesting(timestamp, 
                   pd.Series(all_test_values, index=all_test_dates).loc[common_dates], 
                   pd.Series(all_predictions, index=all_test_dates).loc[common_dates], 
                   returns)

if __name__ == "__main__":
    main() 