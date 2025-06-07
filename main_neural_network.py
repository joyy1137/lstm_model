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

def shuffle_by_sequence_window(X, y, sequence_length):
    n_samples = len(X)
    n_windows = n_samples // sequence_length
    if n_windows == 0:
        return X, y
    
    # 创建窗口索引
    window_indices = np.arange(n_windows)
    np.random.shuffle(window_indices)
    
    # 重新排列数据
    shuffled_X = []
    shuffled_y = []
    
    for window_idx in window_indices:
        start_idx = window_idx * sequence_length
        end_idx = start_idx + sequence_length
        shuffled_X.append(X[start_idx:end_idx])
        shuffled_y.append(y[start_idx:end_idx])
    
    # 处理剩余的数据（如果有的话）
    remaining_start = n_windows * sequence_length
    if remaining_start < n_samples:
        shuffled_X.append(X[remaining_start:])
        shuffled_y.append(y[remaining_start:])
    
    return np.concatenate(shuffled_X), np.concatenate(shuffled_y)

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
        fold_predictions = []
        fold_models = []
        best_val_acc = 0
        best_fold = 1
        
        print(f"\n开始 {Config.n_splits} 折交叉验证训练...")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full), 1):
            print(f"\n训练第 {fold} 折...")
            X_train = X_train_full[train_idx]
            y_train = y_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]
            
            # 对训练集按照sequence_length进行打乱
            X_train, y_train = shuffle_by_sequence_window(X_train, y_train, Config.sequence_length)
            
            # 训练模型
            model = trainer.train_model(
                X_train, y_train,
                X_val, y_val,
                discrete_features_train=None,
                discrete_features_val=None,
                use_grid_search=True,
                config_name=f"model_{date_key}_fold_{fold}"
            )
            
            # 保存模型
            model_path = os.path.join(Config.MODELS_DIR, f'model_{date_key}_fold_{fold}.path')
            trainer.save_model(model, model_path)
            
            # 计算当前fold的验证集准确率
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.FloatTensor(y_val).to(device)
                val_pred = model(X_val_tensor)
                val_pred = torch.sigmoid(val_pred)
                val_pred = (val_pred > 0.5).cpu().numpy()
                val_acc = np.mean(val_pred == y_val)
                
                # 更新最佳fold
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_fold = fold
                    # 只保存最佳fold的历史记录
                    trainer.model_histories[f"model_{date_key}"] = trainer.model_histories[f"model_{date_key}_fold_{fold}"]
            
            # 进行测试集预测
            with torch.no_grad():
                # 检查测试集是否为空
                if len(X_test) == 0:
                    continue
                
                # 将测试数据移动到正确的设备上
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_test_tensor = torch.FloatTensor(y_test).to(device)
                
                # 使用训练好的模型进行预测
                model.to(device)  # 确保模型在正确的设备上
                
                y_test_pred = model(X_test_tensor)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_test_pred = y_test_pred.cpu().numpy()
                y_test_pred = y_test_pred.reshape(-1)  # 确保是1维数组
                y_test_pred = (y_test_pred > 0.5).astype(int)
                
                # 计算并打印测试集准确率
                test_acc = np.mean(y_test_pred == y_test)
                print(f"Fold {fold} 测试集准确率: {test_acc:.4f}")
                
                fold_predictions.append(y_test_pred)
                fold_models.append(model)
        
        print(f"\n最佳验证集准确率: {best_val_acc:.4f} (第 {best_fold} 折)")
        
        # 对所有fold的预测结果进行投票
        if fold_predictions:
            # 将预测结果转换为numpy数组
            fold_predictions = np.array(fold_predictions)
            # 使用多数投票
            final_predictions = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), 
                axis=0, 
                arr=fold_predictions
            )
            
            # 收集预测结果
            all_predictions.extend(final_predictions)
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
    


    
    # 创建对齐的Series
    actual_series = pd.Series(all_test_values, index=all_test_dates)
    predicted_series = pd.Series(all_predictions, index=all_test_dates)
    
    # 处理重复的日期，保留最后一个预测值
    actual_series = actual_series[~actual_series.index.duplicated(keep='last')]
    predicted_series = predicted_series[~predicted_series.index.duplicated(keep='last')]
    
    # 确保所有数据都使用相同的索引
    test_results = pd.DataFrame({
        'valuation_date': common_dates,
        'actual': actual_series.reindex(common_dates),
        'predicted': predicted_series.reindex(common_dates)
    })
   
    print("\n预测准确率:", np.mean(test_results['actual'] == test_results['predicted']))
    
    # 运行回测
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_backtesting(timestamp, 
                   actual_series.reindex(common_dates), 
                   predicted_series.reindex(common_dates), 
                   returns)

if __name__ == "__main__":
    main() 