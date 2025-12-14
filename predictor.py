import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

class FinancialPredictor:
    def __init__(self, model_type: str = "random_forest"):
        """
        初始化财务预测器
        
        Args:
            model_type: 模型类型，支持 "random_forest" 或 "linear_regression"
        """
        self.model_type = model_type
        self.model = None
        self.sequence_length = 4  # 使用4个季度的历史数据进行预测
    
    def _create_sequences(self, data: pd.Series, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据
        
        Args:
            data: 历史数据序列
            seq_length: 序列长度
        
        Returns:
            X: 特征序列
            y: 目标值
        """
        X = []
        y = []
        
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i:i+seq_length].values)
            y.append(data.iloc[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def predict_pe(self, pe_series: pd.Series, forecast_quarters: int = 2) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        预测未来市盈率（按季度）
        
        Args:
            pe_series: 历史市盈率序列
            forecast_quarters: 预测季度数
        
        Returns:
            forecast_df: 包含预测结果的DataFrame
            metrics: 预测模型的评估指标
        """
        # 如果数据是按天的，先转换为季度数据（取每个季度最后一天的数据）
        if pe_series.index.freq is None or pe_series.index.freq < pd.DateOffset(months=3):
            # 按季度重采样，取每个季度最后一个值
            quarterly_pe = pe_series.resample('Q').last()
        else:
            quarterly_pe = pe_series
        
        # 灵活调整序列长度，如果历史数据不足，使用可用的最大长度
        available_sequence_length = len(quarterly_pe) - 1  # 至少需要1个数据点来创建序列
        if available_sequence_length < 1:
            raise ValueError("历史数据不足，无法进行预测")
        
        # 使用可用的最大序列长度，但不超过设定的最大值
        actual_sequence_length = min(self.sequence_length, available_sequence_length)
        
        # 准备数据
        X, y = self._create_sequences(quarterly_pe, actual_sequence_length)
        
        # 训练模型
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X, y)
        
        # 预测未来值
        last_sequence = quarterly_pe.iloc[-actual_sequence_length:].values.reshape(1, -1)
        forecast = []
        
        for _ in range(forecast_quarters):
            next_pred = self.model.predict(last_sequence)[0]
            forecast.append(next_pred)
            
            # 更新序列，用于下一步预测
            last_sequence = np.roll(last_sequence, -1)  # 滚动序列
            last_sequence[0, -1] = next_pred  # 更新最后一个值
        
        # 创建预测索引（按季度）
        last_date = quarterly_pe.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=forecast_quarters, freq="Q")
        
        # 创建预测结果DataFrame
        forecast_df = pd.DataFrame({
            "date": forecast_index,
            "Forecasted_PE": forecast
        }).set_index("date")
        
        # 计算模型评估指标
        predictions = self.model.predict(X)
        metrics = {
            "mean_absolute_error": mean_absolute_error(y, predictions),
            "r2_score": r2_score(y, predictions)
        }
        
        return forecast_df, metrics

    def predict_stock_price(self, price_series: pd.Series, forecast_days: int = 10, forecast_type: str = "daily") -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        预测未来股票价格
        
        Args:
            price_series: 历史股价序列（收盘价）
            forecast_days: 预测天数或季度数
            forecast_type: 预测类型，支持 "daily"（按天）或 "quarterly"（按季度）
        
        Returns:
            forecast_df: 包含预测结果的DataFrame
            metrics: 预测模型的评估指标
        """
        # 根据预测类型处理数据
        if forecast_type == "quarterly":
            # 按季度重采样，取每个季度最后一个值
            processed_data = price_series.resample('Q').last()
            freq = "Q"
            offset = pd.DateOffset(months=3)
        else:
            # 使用每日数据
            processed_data = price_series
            freq = "D"
            offset = pd.DateOffset(days=1)
        
        # 灵活调整序列长度，如果历史数据不足，使用可用的最大长度
        available_sequence_length = len(processed_data) - 1  # 至少需要1个数据点来创建序列
        if available_sequence_length < 1:
            raise ValueError("历史数据不足，无法进行预测")
        
        # 使用可用的最大序列长度，但不超过设定的最大值
        actual_sequence_length = min(self.sequence_length, available_sequence_length)
        
        # 准备数据
        X, y = self._create_sequences(processed_data, actual_sequence_length)
        
        # 训练模型
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X, y)
        
        # 预测未来值
        last_sequence = processed_data.iloc[-actual_sequence_length:].values.reshape(1, -1)
        forecast = []
        
        for _ in range(forecast_days):
            next_pred = self.model.predict(last_sequence)[0]
            forecast.append(next_pred)
            
            # 更新序列，用于下一步预测
            last_sequence = np.roll(last_sequence, -1)  # 滚动序列
            last_sequence[0, -1] = next_pred  # 更新最后一个值
        
        # 创建预测索引
        last_date = processed_data.index[-1]
        forecast_index = pd.date_range(start=last_date + offset, periods=forecast_days, freq=freq)
        
        # 创建预测结果DataFrame
        forecast_df = pd.DataFrame({
            "date": forecast_index,
            "Forecasted_Price": forecast
        }).set_index("date")
        
        # 计算模型评估指标
        predictions = self.model.predict(X)
        metrics = {
            "mean_absolute_error": mean_absolute_error(y, predictions),
            "r2_score": r2_score(y, predictions)
        }
        
        return forecast_df, metrics
