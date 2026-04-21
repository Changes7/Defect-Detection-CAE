import sqlite3
import pandas as pd
import os
import datetime
import torch
import streamlit as st
from model import ConvAutoEncoder

DB_NAME = "defect.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据库写入函数
def log_to_db(prod, defect, mse):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS detection_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, product_type TEXT, result TEXT, defect_count INTEGER, mse REAL)')
    res_text = "❌ 不合格" if defect > 0 else "✅ 合格"
    cursor.execute('INSERT INTO detection_logs (timestamp, product_type, result, defect_count, mse) VALUES (?, ?, ?, ?, ?)', 
                   (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prod, res_text, defect, mse))
    conn.commit()
    conn.close()

# 2. 数据库读取函数 (带缓存)
@st.cache_data(ttl=5)
def get_all_data():
    if not os.path.exists(DB_NAME): return pd.DataFrame()
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM detection_logs ORDER BY id DESC", conn)
    conn.close()
    return df

# 3. 模型加载函数 (带缓存)
@st.cache_resource 
def load_model(path): 
    if not os.path.exists(path): return None
    model = ConvAutoEncoder().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model