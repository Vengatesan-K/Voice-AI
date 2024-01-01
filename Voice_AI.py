#!pip install --upgrade datasets transformers accelerate soundfile librosa evaluate jiwer tensorboard gradio
#pip install transformers[torch]
#pip install accelerate -U
#pip install accelerate==0.20.3
#!pip install transformers

from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import gradio
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset



# Authenticate with Hugging Face Hub
notebook_login()

# Loading the dataset
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "mr", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "mr", split="test", use_auth_token=True)

# Preprocessing the dataset
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Initialize feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Marathi", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Marathi", task="transcribe")

# Function to prepare dataset batches
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Map the preparation function to the dataset
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

# Define data collator for Seq2Seq training
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Padding inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Function to compute evaluation metrics
def compute_metrics(pred):
    # Metrics computation for evaluation
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * evaluate.load("wer").compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Initialize Seq2Seq training arguments
training_args = Seq2SeqTrainingArguments(
    training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=100,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=50,
    save_steps=25,
    eval_steps=25,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
)

# Check GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('CUDA is available.')

# Initialize and train the Seq2Seq model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train()

# Push the trained model to the Hub
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  
    "dataset_args": "config: mr, split: test",
    "language": "hi",
    "model_name": "Whisper Small Hi - Vengatesan",  
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}
trainer.push_to_hub(**kwargs)

# Load the trained model and processor for demonstration
model = WhisperForConditionalGeneration.from_pretrained("Vengatesan/whisper-small-hi")
processor = WhisperProcessor.from_pretrained("Vengatesan/whisper-small-hi")

# Set up the demo interface
def transcribe(audio):
    text = pipeline(model="Vengatesan/whisper-small-hi")(audio)["text"]
    return text

iface = gradio.Interface(
    fn=transcribe,
    inputs=gradio.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Marathi",
    description="Realtime demo for Marathi speech recognition using a fine-tuned Whisper small model.",
)
iface.launch()



### To Evaluate 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

class ASREvaluationMetrics:
    def __init__(self):
        pass

    @staticmethod
    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Initialize the matrix
        distance = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        # Fill in the first row and column
        for i in range(len(ref_words) + 1):
            distance[i][0] = i
        for j in range(len(hyp_words) + 1):
            distance[0][j] = j

        # Compute the Levenshtein distance
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    cost = 0
                else:
                    cost = 1
                distance[i][j] = min(distance[i - 1][j] + 1,
                                     distance[i][j - 1] + 1,
                                     distance[i - 1][j - 1] + cost)

        return float(distance[len(ref_words)][len(hyp_words)]) / len(ref_words)

    @staticmethod
    def cer(reference, hypothesis):
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)

        # Calculate Levenshtein distance
        distance = sum(1 for i, j in zip(ref_chars, hyp_chars) if i != j)

        return float(distance) / len(ref_chars)

    @staticmethod
    def ter(reference, hypothesis):
        # Calculate Token Error Rate (TER)
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()

        num_errors = sum(1 for i, j in zip(ref_tokens, hyp_tokens) if i != j)

        return float(num_errors) / len(ref_tokens)

    @staticmethod
    def ser(reference, hypothesis):
        # Calculate Sentence Error Rate (SER)
        return 1.0 if reference != hypothesis else 0.0

evaluator = ASREvaluationMetrics()

wer = evaluator.wer(reference, result["text"][0])
cer = evaluator.cer(reference, result["text"][0])
ter = evaluator.ter(reference, result["text"][0])
ser = evaluator.ser(reference, result["text"][0])
print(f"WER: {wer}")
print(f"CER: {cer}")
print(f"TER: {ter}")
print(f"SER: {ser}")