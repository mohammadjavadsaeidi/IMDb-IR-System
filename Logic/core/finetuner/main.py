# Instantiate the class
from Logic.core.finetuner.BertFinetuner_mask import BERTFinetuner

print("bert finetuner starting...")

bert_finetuner = BERTFinetuner('/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/IMDB_crawled.json',
                               top_n_genres=5)
print("bert finetuner initialized")

# Load the dataset
bert_finetuner.load_dataset()
print("load data set successfully", bert_finetuner)

# Preprocess genre distribution
bert_finetuner.preprocess_genre_distribution()
print("preprocess genres successfully")

# Split the dataset
bert_finetuner.split_dataset()
print("split data set successfully")

# Fine-tune BERT model
bert_finetuner.fine_tune_bert()
print("fine-tune bert model successfully")

# Compute metrics
print(bert_finetuner.evaluate_model())
print("evaluate model successfully")

# Save the model (optional)
bert_finetuner.save_model('Movie_Genre_Classifier')
print("save model successfully")

# you can see model on https://huggingface.co/datasets/mjsaeidi/MIR_Bert_Finetune/tree/main
# some images saved on output directory
