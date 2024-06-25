import json
import torch
import pandas as pd
import numpy as np
from collections import Counter
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, login
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.top_n_genres)

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        self.df = pd.DataFrame(self.data).dropna(subset=['first_page_summary', 'genres'])

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres.
        """
        genre_counter = Counter(genre for genres in self.df['genres'] for genre in genres)
        self.top_genres = [genre for genre, _ in genre_counter.most_common(self.top_n_genres)]

        def filter_genres(genres):
            return [genre for genre in genres if genre in self.top_genres]

        self.df['filtered_genres'] = self.df['genres'].apply(filter_genres)
        self.df = self.df[self.df['filtered_genres'].map(len) > 0]

        genre_to_id = {genre: idx for idx, genre in enumerate(self.top_genres)}
        self.df['label'] = self.df['filtered_genres'].apply(lambda genres: genre_to_id[genres[0]])

    def split_dataset(self):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        train_df, temp_df = train_test_split(self.df, test_size=0.2, stratify=self.df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

        self.train_encodings = self.tokenizer(train_df['first_page_summary'].tolist(), truncation=True, padding=True)
        self.train_labels = train_df['label'].tolist()

        self.val_encodings = self.tokenizer(val_df['first_page_summary'].tolist(), truncation=True, padding=True)
        self.val_labels = val_df['label'].tolist()

        self.test_encodings = self.tokenizer(test_df['first_page_summary'].tolist(), truncation=True, padding=True)
        self.test_labels = test_df['label'].tolist()

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        train_dataset = self.create_dataset(self.train_encodings, self.train_labels)
        val_dataset = self.create_dataset(self.val_encodings, self.val_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        metric_accuracy = load_metric("accuracy")
        metric_precision = load_metric("precision")
        metric_recall = load_metric("recall")
        metric_f1 = load_metric("f1")

        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
        precision = metric_precision.compute(predictions=predictions, references=labels, average="weighted")
        recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"]
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        test_dataset = self.create_dataset(self.test_encodings, self.test_labels)
        trainer = Trainer(model=self.model)
        return trainer.evaluate(test_dataset)

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)
        print(f"Model saved locally to {model_name}")

        # Push to Hugging Face Model Hub
        print("Pushing model to the Hugging Face Model Hub...")
        login(token="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")
        create_repo(model_name, exist_ok=True)
        repo_url = f"https://huggingface.co/{model_name}"
        repo = Repository(model_name, clone_from=repo_url)
        repo.git_add()
        repo.git_commit("Initial commit")
        repo.git_push()
        print(f"Model pushed to the Hugging Face Model Hub: {repo_url}")
        repo = Repository(model_name, clone_from=repo_url)
        repo.git_add()
        repo.git_commit("Initial commit")
        repo.git_push()
        print(f"Model pushed to the Hugging Face Model Hub: {repo_url}")


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)