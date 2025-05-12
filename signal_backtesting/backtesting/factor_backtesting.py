import os
import pandas as pd
import numpy as np
from signal_backtesting.backtesting.backtesting_tools import Back_testing_processing
from config import Config
class factor_backtesting_main:
    def __init__(self,signal_name,start_date,end_date,cost,inputpath):
        self.df_index_return=self.index_return_withdraw()
        self.signal_name=signal_name
        self.start_date=start_date
        self.end_date=end_date
        self.cost=Config.BACKTESTING_COST
        self.inputpath=inputpath
    def index_return_withdraw(self):
        df_return = pd.read_csv(Config.INDEX_RETURN_PATH,encoding='gbk')
        df_return.columns=['valuation_date','sz50','hs300','zz500','zz1000','zz2000','zzA500','gz2000']
        df_return = df_return[['valuation_date', 'hs300', 'zz2000']]
        df_return['hs300'] = df_return['hs300'].astype(float)
        df_return['zz2000'] = df_return['zz2000'].astype(float)
       
        return df_return

    def raw_signal_withdraw(self):
        try:
            df = pd.read_excel(self.inputpath)
        except:
            df=pd.read_csv(self.inputpath)
        df.columns=['valuation_date','final_signal']
        df['valuation_date']=pd.to_datetime(df['valuation_date'])
        df['valuation_date']=df['valuation_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df.dropna(inplace=True)
        return df
    def probability_processing(self,df_signal):
        df_index = self.index_return_withdraw()
        df_signal = df_signal.merge(df_index, on='valuation_date', how='left')
        df_final=pd.DataFrame()
        df_signal['target'] = df_signal['hs300'] - df_signal['zz2000']
        df_signal.loc[df_signal['target'] > 0, ['target']] = 0
        df_signal.loc[df_signal['target'] < 0, ['target']] = 1
        df_signal['target'] = df_signal['target'].shift(-1)
        df_signal.dropna(inplace=True)
        number_0 = len(df_signal[df_signal['final_signal'] == 0])
        number_1 = len(df_signal[df_signal['final_signal'] == 1])
        number_0_correct = len(df_signal[(df_signal['final_signal'] == 0) & (df_signal['target'] == 0)])
        number_1_correct = len(df_signal[(df_signal['final_signal'] == 1) & (df_signal['target'] == 1)])
        if number_0==0:
            number_0=1
        if number_1==0:
            number_1=1
        pb_0_correct = number_0_correct / number_0
        pb_0_wrong = 1 - pb_0_correct
        pb_1_correct=number_1_correct/number_1
        pb_1_wrong=1-pb_1_correct
        df_final['hs300']=[pb_0_correct,pb_0_wrong]
        df_final['zz2000']=[pb_1_correct,pb_1_wrong]
        return df_final

    def signal_return_processing(self,df_signal,index_name):
        df_index = self.index_return_withdraw()
        df_index['大小盘等权']=0.5*df_index['hs300']+0.5*df_index['zz2000']
        df_signal = df_index.merge(df_signal, on='valuation_date', how='left')
        df_signal.dropna(inplace=True)
        
        # Add probability threshold (0.6) for signal generation
        df_signal['signal_return'] = 0
        df_signal.loc[df_signal['final_signal'] == 0, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 0]['hs300'].tolist()
        df_signal.loc[df_signal['final_signal'] == 1, ['signal_return']] = \
        df_signal.loc[df_signal['final_signal'] == 1]['zz2000'].tolist()
        
        # Handle neutral signals (0.5) based on index_name
        if index_name=='hs300':
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['hs300'].tolist()
        elif index_name=='zz2000':
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['zz2000'].tolist()
        else:
            df_signal.loc[df_signal['final_signal'] == 0.5, ['signal_return']] = \
                df_signal.loc[df_signal['final_signal'] == 0.5]['大小盘等权'].tolist()
        
        # Calculate turnover with minimum holding period
        df_signal['turn_over'] = df_signal['final_signal'] - df_signal['final_signal'].shift(1)
        df_signal['turn_over'] = abs(df_signal['turn_over']) * 2
        
        # Add minimum holding period (5 days)
        min_holding_period = 5
        df_signal['days_since_last_trade'] = df_signal['turn_over'].notna().astype(int).groupby(
            (df_signal['turn_over'].notna().astype(int).cumsum())
        ).cumsum()
        
        # Only allow trading after minimum holding period
        df_signal.loc[df_signal['days_since_last_trade'] < min_holding_period, 'turn_over'] = 0
        
        df_signal.fillna(method='ffill',inplace=True)
        df_signal.fillna(method='bfill',inplace=True)
        
        # Apply transaction cost
        df_signal['turn_over'] = df_signal['turn_over'] * self.cost
        
        # Calculate portfolio returns
        df_signal['portfolio'] = df_signal['signal_return'].astype(float) - df_signal['turn_over']
        df_signal = df_signal[['valuation_date', 'portfolio',index_name]]
        df_signal.rename(columns={index_name:'index'},inplace=True)
        return df_signal
    def backtesting_main(self,outputpath):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath, exist_ok=True)
        bp = Back_testing_processing(self.df_index_return)
        df_signal=self.raw_signal_withdraw()
        df_prob=self.probability_processing(df_signal)
        outputpath_prob=os.path.join(outputpath,'prob_matrix.xlsx')
        df_prob.to_excel(outputpath_prob,index=False)
        for index_name in ['hs300','zz2000','大小盘等权']:
            if index_name=='大小盘等权':
                index_type='combine'
            else:
                index_type='single'
            outputpath_single=os.path.join(outputpath,index_name)
            if not os.path.exists(outputpath_single):
                os.makedirs(outputpath_single, exist_ok=True)
            df_portfolio = self.signal_return_processing(df_signal, index_name)
            bp.back_testing_history(df_portfolio, outputpath_single, index_type, index_name, self.signal_name)


