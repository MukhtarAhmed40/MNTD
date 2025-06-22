import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import pandas as pd
import os

def load_dataset(data_path: str, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
    """
    Load and preprocess dataset from the given path
    Returns: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    # Check if preprocessed data exists
    if os.path.exists(os.path.join(data_path, 'preprocessed_data.npz')):
        return _load_preprocessed_data(data_path)
    
    # Otherwise load raw data and preprocess
    print("Preprocessing data...")
    if 'CICIDS2017' in data_path:
        return _load_cicids2017(data_path, test_size, val_size)
    elif 'DoHBrw2020' in data_path:
        return _load_dohbrw2020(data_path, test_size, val_size)
    else:
        raise ValueError("Unknown dataset. Please provide path to CICIDS2017 or CIRA-CIC-DoHBrw2020")

def _load_preprocessed_data(data_path: str) -> Tuple:
    """Load preprocessed data from .npz file"""
    data = np.load(os.path.join(data_path, 'preprocessed_data.npz'))
    return (
        (data['X_train'], data['y_train']),
        (data['X_val'], data['y_val']),
        (data['X_test'], data['y_test'])
    )

def _load_cicids2017(data_path: str, test_size: float, val_size: float) -> Tuple:
    """Load and preprocess CICIDS2017 dataset"""
    # This is a simplified version - actual implementation would process CSV files
    print("Loading CICIDS2017 data...")
    
    # Simulate loading data (replace with actual CSV loading)
    num_samples = 10000
    num_packets = 30
    bytes_per_packet = 64
    
    # Generate synthetic features and labels
    X = np.random.randint(0, 256, size=(num_samples, num_packets, bytes_per_packet)).astype('float32')
    y = np.random.randint(0, 2, size=(num_samples,))
    
    # Normalize byte values to [0, 1]
    X = X / 255.0
    
    # Split into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train)
    
    # Save preprocessed data for faster loading next time
    np.savez(
        os.path.join(data_path, 'preprocessed_data.npz'),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _load_dohbrw2020(data_path: str, test_size: float, val_size: float) -> Tuple:
    """Load and preprocess CIRA-CIC-DoHBrw2020 dataset"""
    print("Loading CIRA-CIC-DoHBrw2020 data...")
    
    # Simulate loading data (replace with actual CSV loading)
    num_samples = 10000
    num_packets = 30
    bytes_per_packet = 64
    
    # Generate synthetic features and labels
    X = np.random.randint(0, 256, size=(num_samples, num_packets, bytes_per_packet)).astype('float32')
    y = np.random.randint(0, 2, size=(num_samples,))
    
    # Normalize byte values to [0, 1]
    X = X / 255.0
    
    # Split into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train)
    
    # Save preprocessed data for faster loading next time
    np.savez(
        os.path.join(data_path, 'preprocessed_data.npz'),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
