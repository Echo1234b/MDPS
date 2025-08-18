import pandas as pd
import numpy as np
from typing import Tuple, List

class SequenceModeling:
    def __init__(self):
        pass
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data.iloc[i:(i + sequence_length)].values
            target = data.iloc[i + sequence_length][target_column]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def positional_encoding(self, seq_length: int, d_model: int) -> np.ndarray:
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
