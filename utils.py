"""
Utility functions for BERTune.
"""

import os
import json
import pandas as pd
from datasets import Dataset


def load_dataset_from_file(file_path, data_format="csv", text_column=None, label_column=None):
    """
    Load a dataset from a file (CSV or JSON).
    
    Args:
        file_path (str): Path to the dataset file.
        data_format (str): Format of the dataset file ('csv' or 'json').
        text_column (str, optional): Name of the column containing the text data.
        label_column (str, optional): Name of the column containing the labels.
        
    Returns:
        datasets.Dataset: A Hugging Face Dataset object.
    """
 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
   
    if data_format.lower() == "csv":
        df = pd.read_csv(file_path)
    elif data_format.lower() == "json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Supported formats: csv, json")
    
   
    if text_column is None:
        
        text_lengths = {col: df[col].astype(str).str.len().mean() for col in df.columns}
        text_column = max(text_lengths, key=text_lengths.get)
        print(f"Auto-detected text column: {text_column}")
    
    if label_column is None and len(df.columns) > 1:
        
        label_candidates = [col for col in df.columns if col != text_column]
        if label_candidates:
            unique_counts = {col: df[col].nunique() for col in label_candidates}
            label_column = min(unique_counts, key=unique_counts.get)
            print(f"Auto-detected label column: {label_column}")
    
    
    dataset = Dataset.from_pandas(df)
    
    return dataset


def save_labels(labels, output_dir):
    """
    Save label mapping to a JSON file.
    
    Args:
        labels (list or dict): List of labels or label mapping.
        output_dir (str): Directory to save the labels file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(labels, list):
        
        labels_dict = {label: i for i, label in enumerate(labels)}
    else:
        labels_dict = labels
    
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(labels_dict, f, indent=2)


def load_labels(labels_file):
    """
    Load labels from a file.
    
    Args:
        labels_file (str): Path to the labels file.
        
    Returns:
        dict: Label mapping.
    """
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    with open(labels_file, "r") as f:
        labels = json.load(f)
    
    return labels


def get_compute_metrics(task_type):
    """
    Get the appropriate compute_metrics function for the given task type.
    
    Args:
        task_type (str): Type of task ('classification', 'regression', etc.).
        
    Returns:
        function: A function that computes evaluation metrics.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
    import numpy as np
    
    def compute_classification_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def compute_regression_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.squeeze()
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        return {
            'mse': mse,
            'rmse': rmse
        }
    
    if task_type.lower() == 'classification':
        return compute_classification_metrics
    elif task_type.lower() == 'regression':
        return compute_regression_metrics
    else:
        raise ValueError(f"Unsupported task type: {task_type}")