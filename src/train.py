import os
import math
import logging
import requests
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Model
from utils import load_data, prepare_data, get_batch
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

# Hyperparameters with type hints and docstrings
class TrainingConfig:
    """Configuration for model training"""
    BATCH_SIZE: int = 4
    CONTEXT_LENGTH: int = 16
    LEARNING_RATE: float = 1e-3
    MAX_ITERS: int = 5000
    EVAL_INTERVAL: int = 50
    EVAL_ITERS: int = 20
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
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            x_batch, y_batch = get_batch(data, config)
            _, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[f'{split}_loss'] = losses.mean().item()
    
    model.train()
    return out

def main():
    """Main training script"""
    
    # Set random seed for reproducibility
    torch.manual_seed(TrainingConfig.TORCH_SEED)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/sales_llm_experiment')
    
    # 1. Load and prepare data
    raw_data = load_data()
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
    for step in range(TrainingConfig.MAX_ITERS):
        # Periodic evaluation
        if step % TrainingConfig.EVAL_INTERVAL == 0 or step == TrainingConfig.MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, TrainingConfig)
            
            logger.info(
                f'Step {step:>5d}: '
                f'Train Loss = {losses["train_loss"]:>8.4f}, '
                f'Val Loss = {losses["val_loss"]:>8.4f}'
            )
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', losses['train_loss'], step)
            writer.add_scalar('Loss/validation', losses['val_loss'], step)
        
        # Training step
        x_batch, y_batch = get_batch(train_data, TrainingConfig)
        logits, loss = model(x_batch, y_batch)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    logger.info("Training completed")
    
    # 6. Save the model
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model.pth')
    logger.info("Model training completed and saved.")
    
    # Calculate and log the number of parameters
    total_params = calculate_parameters(model)
    logger.info(f"Total parameters: {total_params}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()