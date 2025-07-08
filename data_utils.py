import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy.io
import h5py

class DataPreprocessor:
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
    
    def fit_transform(self, data):
        if self.scaler is not None:
            original_shape = data.shape
            data_reshaped = data.reshape(-1, 1)
            data_scaled = self.scaler.fit_transform(data_reshaped)
            return data_scaled.reshape(original_shape)
        return data
    
    def transform(self, data):
        if self.scaler is not None:
            original_shape = data.shape
            data_reshaped = data.reshape(-1, 1)
            data_scaled = self.scaler.transform(data_reshaped)
            return data_scaled.reshape(original_shape)
        return data
    
    def apply_noise_filter(self, signal, method='moving_average', window_size=5):
        if method == 'moving_average':
            return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        elif method == 'median':
            from scipy.signal import medfilt
            return medfilt(signal, kernel_size=window_size)
        return signal
    
    def segment_signal(self, signal, segment_length=1024, overlap=0.5):
        step = int(segment_length * (1 - overlap))
        segments = []
        
        for i in range(0, len(signal) - segment_length + 1, step):
            segment = signal[i:i + segment_length]
            segments.append(segment)
        
        return np.array(segments)

class DatasetALoader:
    def __init__(self, data_dir, preprocessor=None):
        self.data_dir = data_dir
        self.preprocessor = preprocessor if preprocessor else DataPreprocessor()
        
    def load_ae_data(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.mat':
            data = scipy.io.loadmat(file_path)
            signal = data.get('signal', data.get('data', None))
        elif file_ext == '.h5' or file_ext == '.hdf5':
            with h5py.File(file_path, 'r') as f:
                signal = f['signal'][:]
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            signal = df.values.flatten()
        elif file_ext == '.npy':
            signal = np.load(file_path)
        elif file_ext == '.pkl':
            with open(file_path, 'rb') as f:
                signal = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return signal.astype(np.float32)
    
    def load_dataset(self, segment_length=1024, overlap=0.5):
        signals = []
        labels = []
        metadata = []
        
        valve_types = range(1, 11)
        pressure_levels = [2, 3, 4, 5]
        
        for valve_type in valve_types:
            for pressure in pressure_levels:
                leak_file = os.path.join(self.data_dir, f'valve_{valve_type}_pressure_{pressure}_leak.mat')
                no_leak_file = os.path.join(self.data_dir, f'valve_{valve_type}_pressure_{pressure}_no_leak.mat')
                
                if os.path.exists(leak_file):
                    leak_signal = self.load_ae_data(leak_file)
                    leak_segments = self.preprocessor.segment_signal(leak_signal, segment_length, overlap)
                    
                    for segment in leak_segments:
                        signals.append(segment)
                        labels.append(1)
                        metadata.append({
                            'valve_type': valve_type,
                            'pressure': pressure,
                            'condition': 'leak'
                        })
                
                if os.path.exists(no_leak_file):
                    no_leak_signal = self.load_ae_data(no_leak_file)
                    no_leak_segments = self.preprocessor.segment_signal(no_leak_signal, segment_length, overlap)
                    
                    for segment in no_leak_segments:
                        signals.append(segment)
                        labels.append(0)
                        metadata.append({
                            'valve_type': valve_type,
                            'pressure': pressure,
                            'condition': 'no_leak'
                        })
        
        return np.array(signals), np.array(labels), metadata

class DatasetBLoader:
    def __init__(self, data_dir, preprocessor=None):
        self.data_dir = data_dir
        self.preprocessor = preprocessor if preprocessor else DataPreprocessor()
        
        self.conditions = {
            'OL': 'orifice_leak',
            'LC': 'longitudinal_crack',
            'CC': 'circumferential_crack',
            'GL': 'gasket_leak',
            'NL': 'no_leak'
        }
        
        self.topologies = ['branched', 'looped']
    
    def load_vibration_data(self, file_path):
        return self.load_ae_data(file_path)
    
    def load_dataset(self, segment_length=1024, overlap=0.5):
        signals = []
        labels = []
        metadata = []
        
        for topology in self.topologies:
            for condition_code, condition_name in self.conditions.items():
                file_pattern = os.path.join(self.data_dir, f'{topology}_{condition_name}_*.mat')
                
                import glob
                files = glob.glob(file_pattern)
                
                for file_path in files:
                    signal = self.load_vibration_data(file_path)
                    segments = self.preprocessor.segment_signal(signal, segment_length, overlap)
                    
                    label = 0 if condition_code == 'NL' else 1
                    
                    for segment in segments:
                        signals.append(segment)
                        labels.append(label)
                        metadata.append({
                            'topology': topology,
                            'condition': condition_name,
                            'condition_code': condition_code,
                            'file_path': file_path
                        })
        
        return np.array(signals), np.array(labels), metadata

class PipelineLeakageDataset(Dataset):
    def __init__(self, signals, labels, metadata=None, transform=None):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.metadata = metadata
        self.transform = transform
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        signal = torch.from_numpy(signal).unsqueeze(0)
        
        return signal, label

def create_datasets(data_dir, dataset_type='A', test_size=0.2, 
                   segment_length=1024, overlap=0.5, scaler_type='standard'):
    
    if scaler_type == 'standard':
        preprocessor = DataPreprocessor(scaler_type='standard')
    elif scaler_type == 'minmax':
        preprocessor = DataPreprocessor(scaler_type='minmax')
    else:
        preprocessor = DataPreprocessor(scaler_type='none')

    if dataset_type == 'A':
        loader = DatasetALoader(data_dir, preprocessor=preprocessor)
        all_signals, all_labels, _ = loader.load_dataset(segment_length, overlap)
        
        train_signals, test_signals, train_labels, test_labels = train_test_split(
            all_signals, all_labels, test_size=test_size, random_state=42, stratify=all_labels
        )
        
    elif dataset_type == 'B':
        loader = DatasetBLoader(data_dir, preprocessor=preprocessor)
        all_signals, all_labels, _ = loader.load_dataset(segment_length, overlap)
        
        train_signals, test_signals, train_labels, test_labels = train_test_split(
            all_signals, all_labels, test_size=test_size, random_state=42, stratify=all_labels
        )
    elif dataset_type == 'csv':
        train_file_path = os.path.join(data_dir, 'train.csv')
        test_file_path = os.path.join(data_dir, 'test.csv')

        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Train CSV file not found: {train_file_path}")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test CSV file not found: {test_file_path}")

        train_df = pd.read_csv(train_file_path, header=None)
        test_df = pd.read_csv(test_file_path, header=None)

        if train_df.empty:
            raise ValueError(f"Train CSV file is empty: {train_file_path}")
        if test_df.empty:
            raise ValueError(f"Test CSV file is empty: {test_file_path}")
        
        if train_df.shape[1] < 2:
            raise ValueError(f"Train CSV file must have at least 2 columns (signal + label), but found {train_df.shape[1]} in {train_file_path}")
        if test_df.shape[1] < 2:
            raise ValueError(f"Test CSV file must have at least 2 columns (signal + label), but found {test_df.shape[1]} in {test_file_path}")

        train_signals_raw = train_df.iloc[:, :-1].values
        train_labels = train_df.iloc[:, -1].values
        test_signals_raw = test_df.iloc[:, :-1].values
        test_labels = test_df.iloc[:, -1].values

        train_labels = np.where(train_labels > 0, 1, train_labels)
        test_labels = np.where(test_labels > 0, 1, test_labels)

        train_signals = preprocessor.fit_transform(train_signals_raw)
        test_signals = preprocessor.transform(test_signals_raw)
        
        if segment_length and train_signals.shape[1] != segment_length:
            train_segments = []
            for signal in train_signals:
                train_segments.extend(preprocessor.segment_signal(signal, segment_length, overlap))
            train_signals = np.array(train_segments)
            train_labels = np.repeat(train_labels, [len(preprocessor.segment_signal(s, segment_length, overlap)) for s in train_signals_raw])

        if segment_length and test_signals.shape[1] != segment_length:
            test_segments = []
            for signal in test_signals:
                test_segments.extend(preprocessor.segment_signal(signal, segment_length, overlap))
            test_signals = np.array(test_segments)
            test_labels = np.repeat(test_labels, [len(preprocessor.segment_signal(s, segment_length, overlap)) for s in test_signals_raw])

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    train_dataset = PipelineLeakageDataset(train_signals, train_labels)
    test_dataset = PipelineLeakageDataset(test_signals, test_labels)
    
    print(f"Dataset {dataset_type} loaded successfully:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Total samples (after segmentation): {len(train_dataset) + len(test_dataset)}")
    print(f"  Signal length: {segment_length}")
    
    return train_dataset, test_dataset, preprocessor

def save_datasets(train_dataset, test_dataset, preprocessor, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pt'))
    torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pt'))
    
    with open(os.path.join(save_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Datasets saved to {save_dir}")

def load_datasets(save_dir):
    train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pt'))
    test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pt'))
    
    with open(os.path.join(save_dir, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    return train_dataset, test_dataset, preprocessor

if __name__ == "__main__":
    data_dir = "./datasets/dataset_a"
    
    train_dataset, test_dataset, preprocessor = create_datasets(
        data_dir, 
        dataset_type='A',
        segment_length=1024,
        overlap=0.5,
        scaler_type='standard'
    )
    
    save_datasets(train_dataset, test_dataset, preprocessor, "./processed_datasets")
    
    print("Data processing completed!")