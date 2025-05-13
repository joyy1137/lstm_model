import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from config import Config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = out[:, -1, :]
        
        # 通过全连接层
        out = self.fc(out)
        
        return out

class NeuralNetworkTrainer:
    def __init__(self, random_state=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.model_histories = {}  # Dictionary to store histories of different models
        self.random_state = random_state
        # 设置随机种子
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        np.random.seed(random_state)
    
    def get_trained_model_and_scaler(self):
        """返回训练好的模型"""
        return self.model, None
    
    def train_single_model(self, X_train, y_train, X_val, y_val, params, config_name):
        """训练单个模型并返回验证集性能"""
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device), 
            torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=False
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device), 
            torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params['batch_size'], 
            shuffle=False
        )
        
        # 初始化模型
        input_size = X_train.shape[2]  # 特征维度
        model = LSTMNetwork(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        # 初始化权重
        for name, param in model.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=float(Config.training_parameters['weight_decay'])
        )
        
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=Config.learning_rate_scheduler_parameters['mode'],
            factor=float(Config.learning_rate_scheduler_parameters['factor']),
            patience=int(Config.learning_rate_scheduler_parameters['patience']),
        )
        
        # 训练模型
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = int(Config.training_parameters['patience'])
        patience_counter = 0
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(int(Config.training_parameters['num_epochs'])):
            model.train()
            train_loss = 0
            train_preds = []
            train_true = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 数据增强
                if np.random.random() < Config.augmentation_parameters['augmentation_prob']:
                    # 添加高斯噪声
                    noise = torch.randn_like(batch_X) * Config.augmentation_parameters['noise_scale']
                    batch_X = batch_X + noise
                    
                    # 随机缩放
                    scale = torch.normal(
                        mean=torch.tensor(Config.augmentation_parameters['scale_mean']),
                        std=torch.tensor(Config.augmentation_parameters['scale_std']),
                        size=(batch_X.size(0), 1, 1)
                    ).to(self.device)
                    batch_X = batch_X * scale
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    float(Config.training_parameters['max_norm'])
                )
                
                optimizer.step()
                train_loss += loss.item()
                train_preds.extend((outputs > 0.5).cpu().numpy())
                train_true.extend(batch_y.cpu().numpy())
            
            # 计算训练指标
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_true, train_preds)
            
            # 存储训练指标
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # 验证
            model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    outputs = torch.sigmoid(outputs)
                    val_loss += criterion(outputs, batch_y).item()
                    val_preds.extend((outputs > 0.5).cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_true, val_preds)
            
            # 存储验证指标
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # 恢复最佳模型
                    model.load_state_dict(best_model_state)
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{Config.training_parameters["num_epochs"]}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, '
                      f'Val Acc: {val_acc:.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 存储模型历史记录
        if config_name is not None:
            self.model_histories[config_name] = {
                'train_loss': self.history['train_loss'].copy(),
                'val_loss': self.history['val_loss'].copy(),
                'train_acc': self.history['train_acc'].copy(),
                'val_acc': self.history['val_acc'].copy()
            }
        
        return best_val_loss, model
    
    def train_model(self, X_train, y_train, X_test=None, y_test=None, discrete_features_train=None, 
                   discrete_features_val=None, use_grid_search=True, config_name="default"):
        """训练模型"""
        # 使用TimeSeriesSplit划分训练集和验证集
        tscv = TimeSeriesSplit(n_splits=Config.n_splits)
        for train_idx, val_idx in tscv.split(X_train):
            X_train_final = X_train[train_idx]
            y_train_final = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
        
        if use_grid_search:
            param_grid = Config.param_grid
            best_val_loss = float('inf')
            best_model = None
            best_params = None
            best_history = None
            
            # 生成参数组合
            param_combinations = []
            for param_name, param_config in param_grid.items():
                if param_config['mode'] == 'grid':
                    param_combinations.append(param_config['values'])
                else:  # fixed mode
                    param_combinations.append([param_config['fixed_value']])
            
            # 生成所有参数组合
            param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_combinations)]
            
            print(f"\n开始网格搜索，共 {len(param_combinations)} 种参数组合")
            for i, params in enumerate(param_combinations, 1):
                print(f"\n测试参数组合 {i}/{len(param_combinations)}:")
                print(params)
                
                # 临时训练模型，不保存历史记录
                val_loss, model = self.train_single_model(X_train_final, y_train_final, X_val, y_val, params, None)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_params = params
                    best_history = self.history.copy()  # 保存最佳模型的历史记录
                    print(f"验证损失: {val_loss:.4f}")
            
            print("\n最佳参数组合:")
            print(best_params)
            print(f"最佳验证损失: {best_val_loss:.4f}")
            
            # 直接使用网格搜索中找到的最佳模型
            self.model = best_model
            self.model_histories[config_name] = best_history
        else:
            # 使用默认参数训练模型
            default_params = {
                'hidden_size': 128,
                'num_layers': 1,
                'dropout': 0.3,
                'learning_rate': float(Config.training_parameters['learning_rate']),
                'batch_size': 32
            }
            _, self.model = self.train_single_model(X_train_final, y_train_final, X_val, y_val, default_params, config_name)
        
        return self.model
    
    def save_model(self, model, path):
        """保存模型"""
        torch.save(model.state_dict(), path)
        
    def load_model(self, path):
        """加载模型"""
        model = self.model
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        return model
    
    def plot_all_models_curves(self):
        """Plot accuracy and loss curves for all trained models on separate graphs"""
        if not self.model_histories:
            print("No model histories available to plot")
            return
        
        # Plot accuracy for all models
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        for model_name, history in self.model_histories.items():
            plt.plot(history['train_acc'], label=f'{model_name}', alpha=0.7)
        plt.title('Training Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for model_name, history in self.model_histories.items():
            if any(x is not None for x in history['val_acc']):
                plt.plot(history['val_acc'], label=f'{model_name}', alpha=0.7)
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, 'all_models_accuracy.png'), bbox_inches='tight')
        plt.close()
        
        # Plot loss for all models
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        for model_name, history in self.model_histories.items():
            plt.plot(history['train_loss'], label=f'{model_name}', alpha=0.7)
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for model_name, history in self.model_histories.items():
            if any(x is not None for x in history['val_loss']):
                plt.plot(history['val_loss'], label=f'{model_name}', alpha=0.7)
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, 'all_models_loss.png'), bbox_inches='tight')
        plt.close()
    
    def get_feature_weights(self, model, feature_names, config_name):
        # Get the weights from the first layer
        weights = model.layers[0].weight.data.cpu().numpy()
        
        # Calculate feature importance as the sum of absolute weights
        importance = np.abs(weights).sum(axis=0)
        
        # Create DataFrame with feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Save to CSV
        feature_importance.to_csv(
            os.path.join(Config.FEATURE_WEIGHTS_DIR, f'{config_name}_feature_importance.csv'),
            index=False
        )
        
        return feature_importance 