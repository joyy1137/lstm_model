from signal_backtesting.backtesting.factor_backtesting import factor_backtesting_main
from config import Config
import os
import yaml
import numpy as np
import pandas as pd

def load_paths():
    with open('config/paths.yml', 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)

def evaluate_strategy_performance(y_test, y_test_pred, returns):
    """Evaluate trading strategy performance"""
    returns.columns = ['sz50','hs300','zz500','zz1000','zz2000','zzA500','gz2000']
    strategy_returns = np.where(y_test_pred > 0.5, returns['zz2000'], returns['hs300'])
    benchmark = 0.5 * returns['zz2000'].values + 0.5 * returns['hs300'].values
    excess_returns = strategy_returns - benchmark
    cumulative = np.cumprod(1 + strategy_returns)
    excess_total = np.sum(excess_returns)
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    print(f"\nStrategy Performance:")
    print(f"Total Return: {cumulative[-1] - 1:.4f}")
    print(f"Excess Return: {excess_total:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    return {
        'total_return': float(cumulative[-1] - 1),
        'excess_return': float(excess_total),
        'sharpe_ratio': float(sharpe_ratio)
    }

def run_backtesting(timestamp, y_test=None, y_test_pred=None, returns=None):
    print("\nRunning backtesting...")
    
    # 检查输入数据
    if y_test is None or y_test_pred is None or returns is None:
        print("Missing required data for backtesting. Skipping backtesting step.")
        return
        
    if len(y_test) == 0 or len(y_test_pred) == 0 or len(returns) == 0:
        print("Empty data provided for backtesting. Skipping backtesting step.")
        return
    
    paths = load_paths()
    signal_name = paths['signals']['current_signal']
    inputpath = Config.TEST_PRED_PATH.format(timestamp=timestamp)
    outputpath = paths['files']['backtesting_path'].format(main_folder=paths['main_folder'])
    
    # 创建回测数据
    backtest_data = pd.DataFrame({
        'valuation_date': returns.index,
        'prediction': y_test_pred,
        'actual': y_test
    })
    
    # 保存回测数据
    os.makedirs(os.path.dirname(inputpath), exist_ok=True)
    backtest_data.to_csv(inputpath, index=False)
    
    # 设置回测参数
    start_date = returns.index[0].strftime('%Y-%m-%d')
    end_date = returns.index[-1].strftime('%Y-%m-%d')
    cost = Config.BACKTESTING_COST
    
    # 运行回测
    fbm = factor_backtesting_main(signal_name, start_date, end_date, cost, inputpath)
    outputpath = os.path.join(outputpath, signal_name)
    fbm.backtesting_main(outputpath)
    
    print("Backtesting completed!") 