import os
import math
import logging
import requests
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json

from model import Model
from utils import load_data_with_url, load_data_with_huggingface, prepare_data, get_batch
from tokenizer import TextTokenizer
from parameters import calculate_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("./logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hyperparameters
# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

class TrainingConfig:
    BATCH_SIZE = config['train']['batch_size']
    EPOCHS = config['train']['epochs']
    LEARNING_RATE = config['train']['learning_rate']
    EVAL_INTERVAL = config['train']['eval_interval']
    CONTEXT_LENGTH = config['train']['context_length']
    SEED = config['train']['seed']
    DATASET_PATH = config['train']['dataset_path']
    MODEL_PATH = config['train']['model_path']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """
    Estimate model loss on training and validation sets
    
    Args:
        model (Model): Trained model
        train_data (list): Training data
        val_data (list): Validation data
        config (TrainingConfig): Training configuration
    
    Returns:
        dict: Losses for training and validation sets
    """
    out = {}
    model.eval()
    
    # Estimate loss on training set
    train_loss = 0
    for i in range(0, len(train_data), config.BATCH_SIZE):
        x_batch, y_batch = get_batch(train_data, config)
        _, loss = model(x_batch, y_batch)
        train_loss += loss.item()
    out['train_loss'] = train_loss / len(train_data)
    
    # Estimate loss on validation set
    val_loss = 0
    for i in range(0, len(val_data), config.BATCH_SIZE):
        x_batch, y_batch = get_batch(val_data, config)
        _, loss = model(x_batch, y_batch)
        val_loss += loss.item()
    out['val_loss'] = val_loss / len(val_data)
    
    model.train()
    return out

def main():
    """Main training script"""
    
    # Set random seed for reproducibility
    torch.manual_seed(TrainingConfig.SEED)
    
    # # Initialize TensorBoard writer
    # writer = SummaryWriter('runs/sales_llm_experiment')
    
    # 1. Load and prepare data (parquet format)
    raw_data = load_data_with_huggingface(path=TrainingConfig.DATASET_PATH)
    
    textbook = []
    for i in range(0, len(raw_data['train'])):
        textbook.append(raw_data['train'][i]['text'])
    
    textbook = ' '.join(textbook)
    
    if raw_data is None:
        logger.error("Failed to load training data")
        return
    logger.info(f"Data loaded: {len(textbook)} tokens")
    
    # 2. Tokenize data
    tokenizer = TextTokenizer(encoding_name="cl100k_base")
    tokenized_data = tokenizer.tokenize(textbook)
    train_data, val_data = prepare_data(tokenized_data)   # Split data into train and validation (80/20)
    logger.info(f"Data split: Train {len(train_data)} tokens, Validation {len(val_data)} tokens")
    
    # 3. Initialize model
    model = Model(
    ).to(TrainingConfig.DEVICE)
    logger.info("Model initialized")
    
    # 4. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TrainingConfig.LEARNING_RATE
    )
    logger.info("Optimizer initialized")
    
    # 5. Training loop
    for epoch in range(TrainingConfig.EPOCHS):
        for i in range(0, len(train_data), TrainingConfig.BATCH_SIZE):
            # get a batch of data
            x_batch, y_batch = get_batch(train_data, TrainingConfig)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass and compute loss
            _, loss = model(x_batch, y_batch)

            # backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # log loss at intervals
            if i % TrainingConfig.EVAL_INTERVAL == 0:
                logger.info(
                    f"[Epoch: {epoch:4d}, Iteration: {i:6d}] Loss: {loss.item():8.4f}"
                )
                # Optionally log to tensorboard
                # writer.add_scalar('Loss/train', loss.item(), i)

        # estimate loss on training and validation sets at the end of each epoch
        losses = estimate_loss(model, train_data, val_data, TrainingConfig)
        logger.info(
            f"[Epoch: {epoch:4d}] Train loss: {losses['train_loss']:8.4f} | Validation loss: {losses['val_loss']:8.4f}"
        )

        # Optionally log to tensorboard
        # writer.add_scalars('Loss', losses, epoch)
    logger.info("Training completed")
    
    # 6. Save the model
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), TrainingConfig.MODEL_PATH)
    logger.info("Model training completed and saved.")
    
    # Calculate and log the number of parameters
    total_params = calculate_parameters(model=model, path=TrainingConfig.MODEL_PATH)
    logger.info(f"Total parameters: {total_params}")
    
    # # Close TensorBoard writer
    # writer.close()

if __name__ == '__main__':
    main()