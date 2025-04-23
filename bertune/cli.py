"""
Command-line interface for BERTune.
"""

import os
import logging
import click

try:
    from .trainer import BERTFineTuner
    from .utils import load_dataset_from_file
except ImportError:
    from trainer import BERTFineTuner
    from utils import load_dataset_from_file

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """
    BERTune: A CLI tool for fine-tuning BERT models on custom datasets.
    """
    pass


@cli.command()
@click.option("--dataset", required=True, help="Path to the training dataset (CSV, JSON, etc.).")
@click.option("--validation_dataset", default=None, help="Path to a validation dataset for evaluation during training (CSV, JSON).")
@click.option("--data_format", default="csv", help="Specify the format of the dataset (default: csv, can be json or others).")
@click.option("--text_column", default=None, help="Name of the column containing the text data.")
@click.option("--label_column", default=None, help="Name of the column containing the labels.")

@click.option("--model", required=True, help="Pre-trained model to fine-tune (e.g., bert-base-uncased, bert-large-uncased).")
@click.option("--checkpoint", default=None, help="Path to a saved checkpoint from a previous fine-tuning session to resume training.")

@click.option("--task", required=True, help="Type of task for fine-tuning (e.g., classification, ner, qa, etc.).")
@click.option("--labels", default=None, help="Path to a file containing the label names (e.g., for classification tasks).")

@click.option("--learning_rate", default=2e-5, type=float, help="The learning rate for training (default: 2e-5).")
@click.option("--batch_size", default=16, type=int, help="The batch size for training (default: 16).")
@click.option("--epochs", default=3, type=int, help="The number of epochs to train the model (default: 3).")
@click.option("--warmup_steps", default=0, type=int, help="Number of steps to perform learning rate warmup (default: 0).")
@click.option("--weight_decay", default=0.01, type=float, help="Weight decay for regularization (default: 0.01).")
@click.option("--max_seq_length", default=128, type=int, help="The maximum sequence length for tokenized input (default: 128).")
@click.option("--gradient_accumulation_steps", default=1, type=int, 
              help="Number of steps to accumulate gradients before performing a backward pass (default: 1).")
@click.option("--logging_steps", default=500, type=int, help="Number of steps between logging updates (default: 500).")

@click.option("--device", default=None, help="Specify whether to use GPU or CPU. Possible values: cpu, cuda.")
@click.option("--fp16", is_flag=True, help="Whether to use 16-bit precision (faster training on supported GPUs).")
@click.option("--fp16_opt_level", default="O1", help="Floating point precision level for FP16 (e.g., O1, O2).")

@click.option("--eval_steps", default=None, type=int, help="The number of steps between evaluations (default: 1000).")
@click.option("--evaluation_strategy", default="epoch", 
              type=click.Choice(["steps", "epoch"]), help="Strategy for evaluation during training. Possible values: steps, epoch (default: epoch).")
@click.option("--output_dir", required=True, help="The directory where the fine-tuned model will be saved.")
@click.option("--save_steps", default=500, type=int, help="Number of steps between saving model checkpoints (default: 500).")
@click.option("--save_total_limit", default=3, type=int, help="The maximum number of checkpoints to save (default: 3).")

@click.option("--logging_dir", default=None, help="Directory to save logs for monitoring training.")
@click.option("--tensorboard", is_flag=True, help="Whether to enable TensorBoard logging for training metrics.")
@click.option("--disable_tqdm", is_flag=True, help="Whether to disable the progress bar (TQDM).")
@click.option("--disable_progress_bar", is_flag=True, help="Disable progress bar during training.")
@click.option("--logging_first_step", is_flag=True, help="Whether to log the first step.")

@click.option("--overwrite_output_dir", is_flag=True, help="Whether to overwrite the output directory if it exists (default: False).")
@click.option("--do_train", is_flag=True, default=True, help="Whether to perform training. Should be True for fine-tuning.")
@click.option("--do_eval", is_flag=True, default=True, help="Whether to evaluate the model after training (default: True).")
@click.option("--do_predict", is_flag=True, help="Whether to predict on the validation or test set after training.")

@click.option("--seed", default=42, type=int, help="Random seed for reproducibility (default: 42).")
@click.option("--adam_epsilon", default=1e-8, type=float, help="Epsilon parameter for Adam optimizer (default: 1e-8).")
def finetune(dataset, validation_dataset, data_format, text_column, label_column, model, checkpoint, task,
             labels, learning_rate, batch_size, epochs, warmup_steps, weight_decay, max_seq_length, 
             gradient_accumulation_steps, logging_steps, device, fp16, fp16_opt_level, eval_steps, evaluation_strategy,
             output_dir, save_steps, save_total_limit, logging_dir, tensorboard, disable_tqdm, disable_progress_bar,
             logging_first_step, overwrite_output_dir, do_train, do_eval, do_predict, seed, adam_epsilon):
    """
    Fine-tune a BERT model on a custom dataset.
    """
    import torch
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    if os.path.exists(output_dir) and not overwrite_output_dir:
        logger.warning(f"Output directory {output_dir} already exists. Use --overwrite_output_dir to overwrite.")
    
    logger.info("Initializing BERTFineTuner")
    tuner = BERTFineTuner(
        model_name=model,
        task=task,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        device=device
    )
    
    logger.info("Preparing dataset")
    datasets = tuner.prepare_dataset(
        dataset_path=dataset,
        validation_dataset_path=validation_dataset,
        data_format=data_format,
        text_column=text_column,
        label_column=label_column,
        labels_file=labels
    )
    
    logger.info("Configuring model")
    num_labels = None
    if task.lower() == 'classification' and tuner.label_map:
        num_labels = len(tuner.label_map)
    tuner.configure_model(num_labels=num_labels, checkpoint=checkpoint)
    
    additional_args = {
        "fp16_opt_level": fp16_opt_level,
        "adam_epsilon": adam_epsilon,
        "logging_first_step": logging_first_step,
        "disable_progress_bar": disable_progress_bar or disable_tqdm,
    }
    
    if do_train:
        logger.info("Starting training")
        metrics = tuner.train(
            datasets=datasets,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            fp16=fp16,
            logging_dir=logging_dir,
            tensorboard=tensorboard,
            disable_tqdm=disable_tqdm,
            evaluation_strategy=evaluation_strategy,
            save_total_limit=save_total_limit,
            **additional_args
        )
    
    if do_eval and tuner.trainer is not None:
        logger.info("Evaluating model on validation set")
        eval_metrics = tuner.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
    
    if do_predict and tuner.trainer is not None:
        logger.info("Make predictions on test set is enabled, but no test set is provided.")
        logger.info("Use the 'predict' or 'evaluate' command for predictions on new data.")
    
    logger.info(f"Training completed. Metrics: {metrics}")
    logger.info(f"Model saved to: {output_dir}")


@cli.command()
@click.option("--model_path", required=True, help="Path to the fine-tuned model.")
@click.option("--input_file", required=True, help="Path to a file containing texts to predict.")
@click.option("--output_file", required=True, help="Path to save the predictions.")
@click.option("--device", default=None, help="Specify whether to use GPU or CPU ('cuda' or 'cpu').")
def predict(model_path, input_file, output_file, device):
    """
    Make predictions using a fine-tuned model.
    """
    import json
    
    logger.info(f"Loading model from {model_path}")
    tuner = BERTFineTuner.load_from_pretrained(model_path, device=device)
    
    logger.info(f"Reading input from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info("Making predictions")
    predictions = tuner.predict(texts)
    
    logger.info(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info("Prediction completed")


@cli.command()
@click.option("--model_path", required=True, help="Path to the fine-tuned model.")
@click.option("--test_dataset", required=True, help="Path to the test dataset.")
@click.option("--data_format", default="csv", help="Format of the dataset ('csv' or 'json').")
@click.option("--text_column", default=None, help="Name of the column containing the text data.")
@click.option("--label_column", default=None, help="Name of the column containing the labels.")
@click.option("--output_file", default=None, help="Path to save the evaluation results.")
@click.option("--device", default=None, help="Specify whether to use GPU or CPU ('cuda' or 'cpu').")
def evaluate(model_path, test_dataset, data_format, text_column, label_column, output_file, device):
    """
    Evaluate a fine-tuned model on a test dataset.
    """
    import json
    from .utils import load_dataset_from_file
    
    logger.info(f"Loading model from {model_path}")
    tuner = BERTFineTuner.load_from_pretrained(model_path, device=device)
    
    logger.info(f"Loading test dataset from {test_dataset}")
    test_data = load_dataset_from_file(
        test_dataset,
        data_format=data_format,
        text_column=text_column,
        label_column=label_column
    )
    
    if hasattr(tuner, 'label_map') and tuner.label_map and label_column:
        def convert_labels(example):
            if label_column in example:
                example['label'] = tuner.label_map.get(example[label_column], 0)  # Default to 0 if not found
            return example
        
        test_data = test_data.map(convert_labels)
    
    tokenized_test = tuner._tokenize_dataset(test_data, text_column, 'label')
    
    logger.info("Evaluating model")
    metrics = tuner.evaluate(tokenized_test)
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    if output_file:
        logger.info(f"Saving evaluation results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)


@cli.command()
@click.option("--input_text", required=True, help="Text to classify.")
@click.option("--model_path", required=True, help="Path to the fine-tuned model.")
@click.option("--device", default=None, help="Specify whether to use GPU or CPU ('cuda' or 'cpu').")
def classify(input_text, model_path, device):
    """
    Classify a single text input using a fine-tuned model.
    """
    logger.info(f"Loading model from {model_path}")
    tuner = BERTFineTuner.load_from_pretrained(model_path, device=device)
    
    result = tuner.predict([input_text])
    
    prediction = result["predictions"][0]
    if "probabilities" in result:
        probabilities = result["probabilities"][0]
        if isinstance(probabilities, list):
            if tuner.label_map:
                reverse_label_map = {v: k for k, v in tuner.label_map.items()}
                for i, prob in enumerate(probabilities):
                    label = reverse_label_map.get(i, i)
                    click.echo(f"{label}: {prob:.4f}")
            else:
                for i, prob in enumerate(probabilities):
                    click.echo(f"Class {i}: {prob:.4f}")
        else:
            click.echo(f"Probability: {probabilities:.4f}")
    
    click.echo(f"Prediction: {prediction}")


def main():
    """
    Main entry point for the CLI.
    """
    cli()


if __name__ == "__main__":
    main()