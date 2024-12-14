import os
import math
import logging
import requests
import torch
import torch.optim as optim
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
class TrainingConfig:
    """Configuration for training the model"""
    def __init__(self, config_path: str = 'config.json'):
        """
        Load configuration from JSON file
        
        Args:
            config_path (str): Path to the configuration JSON file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.BATCH_SIZE = config['train']['batch_size']
        self.EPOCHS = config['train']['epochs']
        self.LEARNING_RATE = config['train']['learning_rate']
        self.EVAL_INTERVAL = config['train']['eval_interval']
        self.CONTEXT_LENGTH = config['train']['context_length']
        self.SEED = config['train']['seed']
        self.DATASET_PATH = config['train']['dataset_path']
        self.MODEL_PATH = config['train']['model_path']
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, device):
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
    
    # Estimate loss on training set
    train_loss = 0
    for i in range(0, len(train_data), batch_size):
        x_batch, y_batch = get_batch(train_data, context_length, batch_size, device)
        _, loss = model(x_batch, y_batch)
        train_loss += loss.item()
    out['train_loss'] = train_loss / len(train_data)
    
    # Estimate loss on validation set
    val_loss = 0
    for i in range(0, len(val_data), batch_size):
        x_batch, y_batch = get_batch(val_data, context_length, batch_size, device)
        _, loss = model(x_batch, y_batch)
        val_loss += loss.item()
    out['val_loss'] = val_loss / len(val_data)
    
    model.train()
    return out

def train_model(model, optimizer, train_data, val_data, batch_size, epochs, eval_interval, context_length, device, seed):
    """
    Train the model
    
    Args:
        model (Model): Model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        train_data (list): Training data
        val_data (list): Validation data
        config (TrainingConfig): Training configuration
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(train_data), batch_size):
            # get a batch of data
            x_batch, y_batch = get_batch(train_data, context_length, batch_size, device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass and compute loss
            _, loss = model(x_batch, y_batch)

            # backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # log loss at intervals
            if i % eval_interval == 0:
                logger.info(
                    f"[Epoch: {epoch:4d}, Iteration: {i:6d}] Loss: {loss.item():8.4f}"
                )

        # estimate loss on training and validation sets at the end of each epoch
        losses = estimate_loss(model, train_data, val_data, batch_size, context_length, device)
        logger.info(
            f"[Epoch: {epoch:4d}] Train loss: {losses['train_loss']:8.4f} | Validation loss: {losses['val_loss']:8.4f}"
        )

def main():
    """Main training script"""
    try:
        config = TrainingConfig()
        
        # 1. Load and prepare data (parquet format)
        raw_data = load_data_with_huggingface(path=config.DATASET_PATH)
        
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
        ).to(config.DEVICE)
        logger.info("Model initialized")
        
        # 4. Optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE
        )
        logger.info("Optimizer initialized")
        
        # 5. Training loop
        train_model(model, optimizer, train_data, val_data, config.BATCH_SIZE, config.EPOCHS, config.EVAL_INTERVAL, config.CONTEXT_LENGTH, config.DEVICE, config.SEED)
        logger.info("Training completed")
        
        # 6. Save the model
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), config.MODEL_PATH)
        logger.info("Model training completed and saved.")
        
        # Calculate and log the number of parameters
        total_params = calculate_parameters(model=model, path=config.MODEL_PATH)
        logger.info(f"Total parameters: {total_params}")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == '__main__':
    main()