#!/usr/bin/env python
"""
Training script for a layer-wise interpretable fine-tuning of a GPT-2 model.
This script:
  1. Takes a .tar.gz file as input (the archive contains many .pdf.gz files).
  2. Extracts the archive to a temporary directory.
  3. Recursively finds and decompresses each .pdf.gz file.
  4. Extracts text from each PDF using PyPDF2.
  5. Tokenizes the text into overlapping chunks and builds a Dataset.
  6. Fine-tunes a GPT-2 model with a custom loss combining the standard LM loss
     and an auxiliary loss computed from an intermediate layer.
     
Usage:
  python train.py --tar_file path/to/data.tar.gz --output_dir ./checkpoints
"""

import argparse
import os
import tarfile
import glob
import gzip
import shutil
import logging
import io

from tqdm import tqdm
from bs4 import BeautifulSoup

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
from PyPDF2 import PdfReader  # ensure PyPDF2 is installed (pip install PyPDF2)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# PDF Processing and Dataset
# -------------------------------
def extract_text_from_pdf(pdf_file_obj):
    """
    Extract text from a PDF file-like object.
    """
    text = ""
    try:
        pdf_file_obj.seek(0)  # <<< ADD THIS LINE
        reader = PdfReader(pdf_file_obj)
        for page in reader.pages:
            page_text = page.extract_text()
            if isinstance(page_text, str) and page_text.strip():
                text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
    return text



# ----- NEW FUNCTION TO HANDLE MULTIPLE FORMATS -----
def extract_text_from_gz(gz_path):
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            content_bytes = f_in.read()

        file_stub = os.path.splitext(gz_path)[0].lower()
        file_like = io.BytesIO(content_bytes)

        if file_stub.endswith(".pdf"):
            return extract_text_from_pdf(file_like)

        elif file_stub.endswith(".html") or file_stub.endswith(".htm"):
            html = content_bytes.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()

        elif file_stub.endswith(".txt"):
            return content_bytes.decode("utf-8", errors="ignore")

        elif file_stub.endswith(".ps"):
            logger.warning(f"PS file not supported (skipped): {gz_path}")
            return ""

        else:
            logger.warning(f"Skipping unknown file type: {gz_path}")
            return ""
    except Exception as e:
        logger.error(f"Failed to decompress and read {gz_path}: {e}")
        return ""


def extract_all_pdf_texts(tar_file, extract_dir="extracted_files"):
    """
    Extracts a .tar.gz file to a directory, finds all .pdf.gz files recursively,
    decompresses them, and extracts text from each PDF.
    Returns a list of extracted text strings.
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        logger.info(f"Extracting tar.gz archive: {tar_file}")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    else:
        logger.info(f"Skipping extraction — folder already exists: {extract_dir}")


    # ----- UPDATED to support all .gz types -----
    gz_files = glob.glob(os.path.join(extract_dir, "**", "*.gz"), recursive=True)
    logger.info(f"Found {len(gz_files)} compressed files.")

    texts = []
    for gz_path in tqdm(gz_files, desc="Processing .gz files"):
        text = extract_text_from_gz(gz_path)
        if text.strip():
            texts.append(text)
    return texts

class PDFTextDataset(Dataset):
    """
    A Dataset that takes raw texts from PDFs, tokenizes them,
    and splits them into chunks of fixed length for causal language modeling.
    """
    def __init__(self, texts, tokenizer, max_length=512, stride=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []
        for doc in texts:
            # Clean up the document text
            doc = " ".join(doc.split())
            tokens = tokenizer.encode(doc)
            # Split tokens into overlapping chunks
            for i in range(0, len(tokens), self.stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) < 5:
                    continue  # skip very short sequences
                self.examples.append(chunk)
        logger.info(f"Created {len(self.examples)} training examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long)
        }

# -------------------------------
# Custom Model with Auxiliary Loss
# -------------------------------
class LayerwiseAuxLM(nn.Module):
    """
    Wrapper for a GPT-2 LM head model.
    Adds an auxiliary head on an intermediate layer to compute extra loss.
    """
    def __init__(self, model_name="gpt2", aux_layer_index=6, aux_loss_weight=1.0):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        
        num_layers = self.model.config.n_layer  # e.g., 12 for gpt2
        if aux_layer_index < 0 or aux_layer_index >= num_layers:
            raise ValueError(f"aux_layer_index must be in [0, {num_layers-1}], got {aux_layer_index}")
        self.aux_layer_index = aux_layer_index
        self.aux_loss_weight = aux_loss_weight

        hidden_size = self.model.config.n_embd  # e.g., 768 for gpt2
        vocab_size = self.model.config.vocab_size

        # Define an auxiliary head: a linear projection mapping hidden states to vocab logits.
        self.aux_head = nn.Linear(hidden_size, vocab_size)
        # Optional: tie aux_head weights to input embeddings for consistency.
        self.aux_head.weight = self.model.transformer.wte.weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None, task_type="general"):
        # Get outputs from GPT-2 model; output_hidden_states=True returns a tuple.
        outputs = self.model(input_ids, labels=labels, output_hidden_states=True)
        logits = outputs.logits  # final layer logits
        hidden_states = outputs.hidden_states  # (embedding, layer1, ..., layer_n)
        #jcole
        loss = None
        main_loss = None
        aux_loss = None

        if labels is not None:
            # Shift logits and labels for causal LM loss (ignoring the last token prediction):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Compute auxiliary logits from an intermediate layer.
            aux_hidden = hidden_states[self.aux_layer_index + 1]  # +1 because index 0 is embedding layer.
            aux_logits = self.aux_head(aux_hidden)
            shift_aux_logits = aux_logits[..., :-1, :].contiguous()
            aux_loss = self.ce_loss(shift_aux_logits.view(-1, shift_aux_logits.size(-1)), shift_labels.view(-1))

            # Combine the main and auxiliary losses.
            loss = main_loss + self.aux_loss_weight * aux_loss

        return {
            "loss": loss,
            "main_loss": main_loss,
            "aux_loss": aux_loss,
            "logits": logits
        }

# -------------------------------
# Training Loop
# -------------------------------
def collate_fn(batch):
    """
    Collate function to pad sequences to the maximum length in the batch.
    GPT-2 does not have a pad token; we set it to the end-of-sequence token.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    # Set pad_token to eos_token if not already set
    pad_token_id = 50256
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract text from PDFs in the .tar.gz archive (contains .pdf.gz files)
    logger.info(f"Extracting PDF texts from tar.gz file: {args.tar_file}")
    texts = extract_all_pdf_texts(args.tar_file, extract_dir=args.extract_dir)
    if not texts:
        logger.error("No text extracted from the provided files. Exiting.")
        return

    # Initialize tokenizer (also used by our model later)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create the dataset and dataloader.
    dataset = PDFTextDataset(texts, tokenizer, max_length=args.max_length, stride=args.stride)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize the custom model.
    model = LayerwiseAuxLM(model_name=args.model_name, aux_layer_index=args.aux_layer_index,
                           aux_loss_weight=args.aux_loss_weight)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    model.train()
    global_step = 0
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels, task_type="general")
            loss = outputs["loss"]
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint at the end of the epoch.
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args)
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    logger.info("Training completed.")

# -------------------------------
# Main function and Argument Parsing
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a GPT-2 model with a unified loss (LM loss + auxiliary loss) on PDF texts extracted from a .tar.gz file containing .pdf.gz files.")
    parser.add_argument("--tar_file", type=str, required=True, help="Path to the .tar.gz file containing .pdf.gz files.")
    parser.add_argument("--extract_dir", type=str, default="extracted_files", help="Directory where the tar file will be extracted.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Pre-trained model name (from Hugging Face).")
    parser.add_argument("--aux_layer_index", type=int, default=6, help="Index of the transformer layer used for auxiliary loss (0-indexed from first transformer block).")
    parser.add_argument("--aux_loss_weight", type=float, default=1.0, help="Weight for the auxiliary loss.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for each training example.")
    parser.add_argument("--stride", type=int, default=256, help="Stride for splitting documents into overlapping chunks.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for AdamW optimizer.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)