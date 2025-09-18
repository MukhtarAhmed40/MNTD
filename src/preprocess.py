# src/preprocess.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import yaml

def load_csvs(paths):
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def basic_preprocess(df, numeric_cols, cat_cols):
    # Remove incomplete flows
    df = df.dropna().reset_index(drop=True)
    # Numeric scaling
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(df[numeric_cols].values)
    # One-hot categorical
    if cat_cols:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_cat = ohe.fit_transform(df[cat_cols].values)
    else:
        X_cat = None
    # Combined
    if X_cat is not None:
        X = np.hstack([X_num, X_cat])
    else:
        X = X_num
    y = df['label'].apply(lambda x: 1 if x in ['malicious','attack',1] else 0).values
    return X.astype(np.float32), y.astype(np.int32), scaler, (ohe if cat_cols else None)

def construct_channels(X, y, df_meta, channel_key_cols=['src_ip','dst_ip','dst_port'], time_col='timestamp', window_seconds=30):
    """
    Aggregates flows into channels as described in the paper (flows sharing src,dst,dst_port within time window).
    df_meta is the full df (with timestamps and IPs)
    Returns list of (channel_feature_matrix, channel_label)
    For simplicity this will group by channel_key and sliding time windows.
    """
    channels = []
    grouped = df_meta.groupby(channel_key_cols)
    for _, group in grouped:
        group = group.sort_values(time_col)
        times = pd.to_datetime(group[time_col])
        start_idx = 0
        for i in range(len(group)):
            if (times.iloc[i] - times.iloc[start_idx]).total_seconds() > window_seconds:
                chunk = group.iloc[start_idx:i]
                if len(chunk) > 0:
                    channels.append((chunk.index.values, chunk))
                start_idx += 1
        # final chunk
        chunk = group.iloc[start_idx:]
        if len(chunk) > 0:
            channels.append((chunk.index.values, chunk))
    return channels
