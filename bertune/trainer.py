"""
BERTune Trainer module 
"""

import os
import logging
from typing import Dict, List, Optional, Union
import json

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import IntervalStrategy

import transformers
import inspect

from datasets import Dataset, DatasetDict

try:
    from .utils import load_dataset_from_file, save_labels, load_labels, get_compute_metrics
except ImportError:
    from utils import load_dataset_from_file, save_labels, load_labels, get_compute_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class BERTFineTuner:
    """
    A class for fine-tuning BERT models on custom datasets.
    """
    
    def __init__(self, 
                 model_name: str,
                 task: str,
                 output_dir: str,
                 max_seq_length: int = 128,
                 device: str = None):
        """
        Initialize the BERTFineTuner.
        
        Args:
            model_name (str): Name of the pre-trained model to fine-tune.
            task (str): Type of task for fine-tuning (e.g., 'classification').
            output_dir (str): Directory to save the fine-tuned model.
            max_seq_length (int, optional): Maximum sequence length for tokenized input.
            device (str, optional): Device to use for training ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.task = task.lower()
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.trainer = None
        self.label_map = None
        
    def _tokenize_dataset(self, dataset: Dataset, text_column: str, label_column: Optional[str] = None) -> Dataset:
        """
        Tokenize the dataset using the BERT tokenizer.
        
        Args:
            dataset (Dataset): The dataset to tokenize.
            text_column (str): Name of the column containing the text data.
            label_column (str, optional): Name of the column containing the labels.
            
        Returns:
            Dataset: The tokenized dataset.
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing dataset"
        )
        
        columns_to_keep = ['input_ids', 'attention_mask', 'token_type_ids']
        if label_column:
            columns_to_keep.append('label')
        
        existing_columns = [col for col in columns_to_keep if col in tokenized_dataset.column_names]
        
        tokenized_dataset = tokenized_dataset.select_columns(existing_columns)
        
        return tokenized_dataset
        
    def prepare_dataset(self, 
                        dataset_path: str,
                        validation_dataset_path: Optional[str] = None,
                        data_format: str = "csv",
                        text_column: Optional[str] = None,
                        label_column: Optional[str] = None,
                        labels_file: Optional[str] = None) -> DatasetDict:
        """
        Prepare the dataset for fine-tuning.
        
        Args:
            dataset_path (str): Path to the training dataset.
            validation_dataset_path (str, optional): Path to the validation dataset.
            data_format (str, optional): Format of the datasets ('csv' or 'json').
            text_column (str, optional): Name of the column containing the text data.
            label_column (str, optional): Name of the column containing the labels.
            labels_file (str, optional): Path to a file containing the label names.
            
        Returns:
            DatasetDict: A dictionary containing the prepared datasets.
        """
        logger.info(f"Loading training dataset from {dataset_path}")
        train_dataset = load_dataset_from_file(
            dataset_path, 
            data_format=data_format,
            text_column=text_column,
            label_column=label_column
        )
        
        if text_column is None:
            text_lengths = {col: train_dataset[col].astype(str).str.len().mean() 
                           for col in train_dataset.column_names}
            text_column = max(text_lengths, key=text_lengths.get)
            logger.info(f"Using auto-detected text column: {text_column}")
        
        if label_column is None and self.task == 'classification':
            label_candidates = [col for col in train_dataset.column_names if col != text_column]
            if label_candidates:
                unique_counts = {col: len(set(train_dataset[col])) for col in label_candidates}
                label_column = min(unique_counts, key=unique_counts.get)
                logger.info(f"Using auto-detected label column: {label_column}")
        
        if self.task == 'classification':
            if labels_file and os.path.exists(labels_file):
                logger.info(f"Loading labels from {labels_file}")
                self.label_map = load_labels(labels_file)
            else:
                unique_labels = sorted(set(train_dataset[label_column]))
                self.label_map = {label: i for i, label in enumerate(unique_labels)}
                
                save_labels(self.label_map, self.output_dir)
                logger.info(f"Created and saved label mapping to {os.path.join(self.output_dir, 'labels.json')}")
            
            def convert_labels(example):
                if label_column in example:
                    example['label'] = self.label_map[example[label_column]]
                return example
            
            train_dataset = train_dataset.map(convert_labels)
            
        if validation_dataset_path:
            logger.info(f"Loading validation dataset from {validation_dataset_path}")
            val_dataset = load_dataset_from_file(
                validation_dataset_path,
                data_format=data_format,
                text_column=text_column,
                label_column=label_column
            )
            
            if self.task == 'classification':
                val_dataset = val_dataset.map(convert_labels)
        else:
            logger.info("No validation dataset provided. Splitting training dataset.")
            train_val_split = train_dataset.train_test_split(test_size=0.1)
            train_dataset = train_val_split['train']
            val_dataset = train_val_split['test']
        
        logger.info("Tokenizing datasets")
        tokenized_train = self._tokenize_dataset(train_dataset, text_column, 'label' if self.task == 'classification' else None)
        tokenized_val = self._tokenize_dataset(val_dataset, text_column, 'label' if self.task == 'classification' else None)
        
        datasets = DatasetDict({
            'train': tokenized_train,
            'validation': tokenized_val
        })
        
        return datasets
    
    def configure_model(self, num_labels: int = None, checkpoint: Optional[str] = None):
        """
        Configure the model for fine-tuning.
        
        Args:
            num_labels (int, optional): Number of labels for classification tasks.
            checkpoint (str, optional): Path to a checkpoint to resume training.
        """
        if checkpoint and os.path.exists(checkpoint):
            logger.info(f"Loading model from checkpoint: {checkpoint}")
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        else:
            if self.task == 'classification':
                if num_labels is None:
                    num_labels = len(self.label_map) if self.label_map else 2
                logger.info(f"Loading model for classification with {num_labels} labels")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=num_labels
                )
            else:
                logger.info("Loading default model for sequence classification")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2
                )
                
        self.model.to(self.device)
    
    def train(self,
              datasets: DatasetDict,
              learning_rate: float = 2e-5,
              batch_size: int = 16,
              epochs: int = 3,
              warmup_steps: int = 0,
              weight_decay: float = 0.01,
              gradient_accumulation_steps: int = 1,
              logging_steps: int = 500,
              eval_steps: Optional[int] = None,
              save_steps: int = 500,
              fp16: bool = False,
              fp16_opt_level: Optional[str] = None,
              adam_epsilon: float = 1e-8,
              logging_dir: Optional[str] = None,
              tensorboard: bool = False,
              disable_tqdm: bool = False,
              evaluation_strategy: str = "epoch",
              save_total_limit: int = 3,
               **kwargs):
        """
        Train the model on the prepared datasets.
        
        Args:
            datasets (DatasetDict): The prepared datasets for training and validation.
            learning_rate (float, optional): Learning rate for training.
            batch_size (int, optional): Batch size for training.
            epochs (int, optional): Number of epochs to train.
            warmup_steps (int, optional): Number of steps for learning rate warm-up.
            weight_decay (float, optional): Weight decay for regularization.
            gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients.
            logging_steps (int, optional): Number of steps between logging updates.
            eval_steps (int, optional): Number of steps between evaluations.
            save_steps (int, optional): Number of steps between saving checkpoints.
            fp16 (bool, optional): Whether to use mixed precision training.
            logging_dir (str, optional): Directory for logging.
            tensorboard (bool, optional): Whether to enable TensorBoard logging.
            disable_tqdm (bool, optional): Whether to disable the progress bar.
            evaluation_strategy (str, optional): Strategy for evaluation during training.
            save_total_limit (int, optional): Maximum number of checkpoints to save.
            
        Returns:
            Dict: Training results.
        """
        if eval_steps is None and evaluation_strategy == "steps":
            eval_steps = logging_steps
            
        if logging_dir is None:
            logging_dir = os.path.join(self.output_dir, "logs")
        #Logging
        logger.info("-" * 50)
        logger.info(f"DEBUG: Attempting to use TrainingArguments")
        logger.info(f"DEBUG:   - transformers version: {transformers.__version__}")
        try:
            logger.info(f"DEBUG:   - TrainingArguments imported from: {inspect.getfile(TrainingArguments)}")
            sig = inspect.signature(TrainingArguments.__init__)
            logger.info(f"DEBUG:   - TrainingArguments.__init__ signature parameters: {list(sig.parameters.keys())}")
            if 'evaluation_strategy' in sig.parameters:
                 logger.info("DEBUG:   - 'evaluation_strategy' IS present in signature.")
            else:
                 logger.error("DEBUG:   - 'evaluation_strategy' IS MISSING from signature!")
        except Exception as e:
            logger.error(f"DEBUG: Failed to inspect TrainingArguments: {e}")
        logger.info("-" * 50)
            
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            evaluation_strategy=IntervalStrategy(evaluation_strategy),
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            fp16=fp16,
            adam_epsilon=adam_epsilon,
            disable_tqdm=disable_tqdm,
            load_best_model_at_end=True,
            report_to=["tensorboard"] if tensorboard else [],
        )
        
        compute_metrics = get_compute_metrics(self.task)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("Starting training")
        train_result = self.trainer.train()
        
        logger.info(f"Saving final model to {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        config_path = os.path.join(self.output_dir, "config.json")
        # if not os.path.exists(config_path):
        #     config = {
        #         "model_type": "bert",
        #         "architectures": ["BertForSequenceClassification"]
        #     }
        #     with open(config_path, "w") as f:
        #         json.dump(config, f)

        self.tokenizer.save_pretrained(self.output_dir)
        
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        return metrics
    
    def evaluate(self, test_dataset: Optional[Dataset] = None):
        """
        Evaluate the fine-tuned model.
        
        Args:
            test_dataset (Dataset, optional): Dataset for evaluation.
            
        Returns:
            Dict: Evaluation results.
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first or load a trained model.")
        
        eval_dataset = test_dataset if test_dataset is not None else self.trainer.eval_dataset
        
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided.")
        
        logger.info("Starting evaluation")
        metrics = self.trainer.evaluate(eval_dataset)
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics
    
    def predict(self, texts: List[str]):
        """
        Make predictions using the fine-tuned model.
        
        Args:
            texts (List[str]): List of texts to predict.
            
        Returns:
            Dict: Predictions and probabilities.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train() first or load a trained model.")
        
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.cpu().numpy()
        
        if self.task == 'classification':
            predictions = np.argmax(logits, axis=1)
            probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
            
            if self.label_map:
                reverse_label_map = {v: k for k, v in self.label_map.items()}
                predicted_labels = [reverse_label_map[pred] for pred in predictions]
            else:
                predicted_labels = predictions.tolist()
            
            return {
                "predictions": predicted_labels,
                "probabilities": probabilities.tolist()
            }
        else:
            return {
                "predictions": logits.flatten().tolist()
            }
    
    @classmethod
    def load_from_pretrained(cls, model_path: str, device: str = None):
        """
        Load a fine-tuned BERTFineTuner model from a directory.
        
        Args:
            model_path (str): Path to the directory containing the fine-tuned model.
            device (str, optional): Device to load the model on ('cpu' or 'cuda').
            
        Returns:
            BERTFineTuner: A BERTFineTuner instance with the loaded model.
        """

        # config_path = os.path.join(model_path, "config.json")
        # if not os.path.exists(config_path):
        #     config = {
        #         "model_type": "bert",
        #         "architectures": ["BertForSequenceClassification"]
        #     }
        #     with open(config_path, "w") as f:
        #         json.dump(config, f)
        labels_path = os.path.join(model_path, "labels.json")
        task = "classification" if os.path.exists(labels_path) else "regression"
        
        instance = cls(
            model_name=model_path,  
            task=task,
            output_dir=model_path,
            device=device
        )
        
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        instance.model.to(instance.device)
        
        if os.path.exists(labels_path):
            instance.label_map = load_labels(labels_path)
        
        return instance