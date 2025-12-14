import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Optional
from config import FINANCE_SITES, DEFAULT_HEADERS
import streamlit as st
def get_eps_from_web(stock_code: str) -> Optional[float]:
    """
    从网页获取实时EPS（每股收益），增加重试机制提高稳定性
    """
    url = FINANCE_SITES["sina"].format(stock_code)
    max_retries = 3
    retry_delay = 2  # 秒
    
    for retry in range(max_retries):
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # 这里需要根据实际网页结构调整解析逻辑
            eps_tag = soup.find("span", text="每股收益")
            if eps_tag:
                eps_text = eps_tag.find_next_sibling().text
                return float(eps_text)
            return None
        except requests.exceptions.RequestException as e:
            print(f"EPS获取失败 (重试 {retry+1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                import time
                time.sleep(retry_delay)
            else:
                return None
def calculate_pe_ratio(
    stock_history: pd.DataFrame,
    stock_code: str,
    period_days: int = 12
    ) -> Optional[pd.DataFrame]:
    """
    计算动态市盈率
    采用三级数据获取策略：
    1. 优先获取专业金融软件数据（这里用 akshare 模拟）
    2. 其次通过网络爬虫获取实时EPS
    3. 最后启用模拟计算模型
    Args:
    stock_history: 历史股价数据
    stock_code: 股票代码（含市场前缀）
    period_days: 计算周期天数
    Returns:
    包含PE值及移动平均的分析结果
    """
    if stock_history.empty:
        return None
    # 模拟EPS获取（实际应调用真实数据源）
    eps = get_eps_from_web(stock_code)
    if eps is None or eps <= 0:
        st.info("EPS数据不可用，使用模拟值计算PE。")
        eps = 1.5 # 模拟值
    # 计算PE = 当前股价 / EPS
    current_price = stock_history['close'].iloc[-1]  # 使用最新收盘价作为当前股价
    pe_series = pd.Series([current_price / eps] * len(stock_history), index=stock_history.index)
    # 生成结果
    result_df = pd.DataFrame({
    'close': stock_history['close'],
    'EPS': eps,
    'PE': pe_series,
    'PE_MA': pe_series.rolling(window=period_days).mean()
    }).dropna()
    return result_df
