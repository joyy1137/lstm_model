from signal_backtesting.PDF.PDFCreator import PDFCreator
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
from matplotlib.font_manager import FontProperties

# Try to use Windows-compatible Chinese fonts
try:
    # Try SimSun first (common on Windows)
    font_path = "C:/Windows/Fonts/simsun.ttc"
    heiti_font = FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = heiti_font.get_name()
except Exception:
    try:
        # Try Microsoft YaHei on Windows
        font_path = "C:/Windows/Fonts/msyh.ttc"
        heiti_font = FontProperties(fname=font_path)
        mpl.rcParams['font.family'] = heiti_font.get_name()
    except Exception:
        try:
            # Try STHeiti on macOS
            font_path = "/System/Library/Fonts/STHeiti Light.ttc"
            heiti_font = FontProperties(fname=font_path)
            mpl.rcParams['font.family'] = heiti_font.get_name()
 
        except: 
                # If no Chinese fonts available, use default font
                print("Warning: No Chinese fonts found. Using default font.")
                mpl.rcParams['font.family'] = 'sans-serif'


mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")
class Back_testing_processing:
    def __init__(self,df_index_return):
        self.df_index_return=df_index_return
    def cal_fund_performance2(self, df1, year):  # 计算一些技术指标 输入端为 portfolio_return 和 index_return
        df=df1.copy()
        df.reset_index(inplace=True, drop=True)
        df['ex_return'] = df['return'] - df['index_return']
        annual_returns2 = (((1 + df['ex_return']).cumprod()).tolist()[-1] - 1) * 252 / len(df)
        annual_returns = round(annual_returns2 * 100, 2)
        vol = round(df['ex_return'].std() * np.sqrt(252), 4)
        # 筛选出基金和基准收益率都为正的数据点
        positive_returns = df[(df['return'] > 0) & (df['index_return'] > 0)]
        # 计算上行捕获收益率
        # upside_returns = round(((1 + positive_returns['ex_return']).cumprod().tolist()[-1] - 1) * 252 / len(df) * 100,
        #                        2)
        # 筛选出基金和基准收益率都为负的数据点
        #negative_returns = df[(df['return'] < 0) & (df['index_return'] < 0)]
        # 计算下行捕获收益率
        #down_returns = round(((1 + negative_returns['ex_return']).cumprod().tolist()[-1] - 1) * 252 / len(df) * 100, 2)
        # 计算夏普比率
        sharpe = round(annual_returns2 / vol, 2)
        # 计算信息比率
        try:
            info_ratio = round((((1 + positive_returns['ex_return']).cumprod().tolist()[-1] - 1) * 252 / len(df)) / vol, 2)
        except:
            print('在时间区间内，positive_returns没有足够长度')
            info_ratio=None
        # IR = round(info_ratio,2)
        df['nav_max'] = (1 + df['ex_return']).cumprod().expanding().max()

        def get_max_drawdown_slow(array):
            drawdowns = []
            for i in range(len(array)):
                max_array = max(array[:i + 1])
                drawdown = (max_array - array[i]) / max_array
                drawdowns.append(drawdown)
            return max(drawdowns)

        max_dd_all = round(get_max_drawdown_slow((1 + df['ex_return']).cumprod().tolist()), 4)
        temp_df = pd.DataFrame({
            year: [annual_returns, sharpe, info_ratio, max_dd_all, vol]
        })
        result_df6 = pd.DataFrame()
        result_df6 = pd.concat([result_df6, temp_df], axis=1)
        result_df6.index = ['年化收益(%)', '夏普比率', '信息比率', '最大回撤', '年化标准差(%)']
        result_df6 = result_df6.T
        result_df6.reset_index(inplace=True)
        result_df6.rename(columns={'index': 'year'}, inplace=True)
        return result_df6
    def draw_gapth(self,df, outputpath, title):  # 画折线图
        df.plot()
        plt.title(title)
        plt.ylabel('净值')
        plt.xlabel('时间')
        file_path = os.path.join(outputpath, "{}图.png".format(title))
        plt.savefig(file_path)
        plt.close()
    def portfolio_return_processing(self,index_name,index_type,df_portfolio):
        if index_type=='single':
              df_index = self.df_index_return[['valuation_date', index_name]]
              df_index.rename(columns={index_name: 'index'}, inplace=True)
        else:
              df_index=self.df_index_return
              df_index['index']=self.df_index_return=0.5*self.df_index_return['hs300']+0.5*self.df_index_return['zz2000']
              df_index=df_index[['valuation_date','index']]
        df_index['valuation_date'] = pd.to_datetime(df_index['valuation_date'])
        df_portfolio['valuation_date'] = pd.to_datetime(df_portfolio['valuation_date'])
        df_portfolio.rename(columns={df_portfolio.columns.tolist()[1]:'portfolio'},inplace=True)
        df_backtesting=df_portfolio.merge(df_index,on='valuation_date',how='left')
        df_backtesting.fillna(0, inplace=True)
        df_backtesting['ex_return'] = df_backtesting['portfolio'] - df_backtesting['index']
        df_backtesting['组合净值'] = (1 + df_backtesting['portfolio']).cumprod()
        df_backtesting['超额净值'] = (1 + df_backtesting['ex_return']).cumprod()
        df_backtesting['基准净值'] = (1 + df_backtesting['index']).cumprod()
        df_h = df_backtesting[['valuation_date', 'portfolio', 'index']]
        df_h2 = df_backtesting[['valuation_date', '组合净值', '超额净值', '基准净值']]
        df_h2.fillna(method='ffill', inplace=True)
        return df_h, df_h2
    def PDF_Creator(self,outputpath, df2, df4,signal_name,index_name):  # df2收益率 df3为权重矩阵 df4为净值
        pdf_filename = os.path.join(outputpath,
                                        '{}回测分析报告.pdf'.format(str(signal_name)))
        pdf = PDFCreator(pdf_filename)
        pdf.title('<b>{}因子分析</b>'.format(signal_name))
        pdf.text('参数:{}'.format(str(index_name)))
        pdf.h1('<b>一、因子表现</b>')
        df2['year'] = df2['valuation_date'].apply(lambda x: str(x)[:4])
        result_df = pd.DataFrame()
        for year in df2['year'].unique().tolist():
            df = df2[df2['year'] == year]
            df_result = self.cal_fund_performance2(df, year)
            result_df = pd.concat([result_df, df_result])
        table_data = [result_df.columns.tolist()] + result_df.values.tolist()
        pdf.table(table_data, highlight_first_row=False)
        pdf.h1('<b>回测区间因子表现</b>')
        df_result2 = self.cal_fund_performance2(df2, year='回测区间')
        table_data = [df_result2.columns.tolist()] + df_result2.values.tolist()
        pdf.table(table_data, highlight_first_row=False)
        pdf.h1('<b>二、净值回测图</b>')
        df4['valuation_date'] = pd.to_datetime(df4['valuation_date'])
        df4.set_index('valuation_date', drop=True, inplace=True)
        slice_df1 = df4[['组合净值', '基准净值']]
        slice_df2 = df4['超额净值']
        self.draw_gapth(slice_df1, outputpath, '组合基准对比')
        fig_filename = os.path.join(outputpath, f"组合基准对比图.png")
        pdf.image(fig_filename)
        self.draw_gapth(slice_df2, outputpath, '超额净值')
        fig_filename = os.path.join(outputpath, f"超额净值图.png")
        pdf.image(fig_filename)
        pdf.build()
        return
    def back_testing_history(self,df_portfolio, outpath_optimizer_result2, index_type,index_name,signal_name):  # 计算analyse_type为history的回测function
            outputpath2=outpath_optimizer_result2
            outputpath_huice1 = os.path.join(outputpath2, str(signal_name) + 'return.xlsx')
            outputpath_huice = os.path.join(outputpath2, str(signal_name)+'回测.xlsx')
            df_portfolio.drop(columns='index',inplace=True)
            df_h, df_h2 = self.portfolio_return_processing(index_name,index_type,df_portfolio)
            df_h.to_excel( outputpath_huice1,index=False)
            df_h2.to_excel(outputpath_huice, index=False)  # 储存文件需要自己定义
            df_h.rename(columns={'portfolio': 'return', 'index': 'index_return'}, inplace=True)
            self.PDF_Creator(outputpath=outputpath2, df2=df_h,df4=df_h2, signal_name=signal_name, index_name=index_name)
