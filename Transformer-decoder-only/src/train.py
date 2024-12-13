import os
import math
import logging
import requests
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Model
from utils import load_data_with_url, load_data_with_huggingface, prepare_data, get_batch
from tokenizer import TextTokenizer, tokenize_data
from parameters import calculate_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('./logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hyperparameters
class TrainingConfig:
    """Configuration for model training"""
    BATCH_SIZE: int = 4
    CONTEXT_LENGTH: int = 16
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 10
    EVAL_INTERVAL: int = 500
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_SEED: int = 1337


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
    torch.manual_seed(TrainingConfig.TORCH_SEED)
    
    # # Initialize TensorBoard writer
    # writer = SummaryWriter('runs/sales_llm_experiment')
    
    # 1. Load and prepare data
    raw_data = str(load_data_with_huggingface()['train']['text'])
    if raw_data is None:
        logger.error("Failed to load training data")
        return
    logger.info(f"Data loaded: {len(raw_data)} tokens")
    
    # 2. Tokenize data
    tokenizer = TextTokenizer(encoding_name="cl100k_base")
    tokenized_data = tokenizer.tokenize(raw_data)
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
    torch.save(model.state_dict(), './model/model.pth')
    logger.info("Model training completed and saved.")
    
    # Calculate and log the number of parameters
    total_params = calculate_parameters(model)
    logger.info(f"Total parameters: {total_params}")
    
    # # Close TensorBoard writer
    # writer.close()

if __name__ == '__main__':
    main()