import torch
import logging
from model import Model
from tokenizer import TextTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model_path, device=None):
        """
        Initialize text generator
        
        Args:
            model_path (str): Path to saved model weights
            device (str, optional): Computing device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = Model().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            self.tokenizer = TextTokenizer()
            logger.info("Model and tokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def generate_text(self, 
                      prompt="", 
                      max_tokens=100, 
                      temperature=0.7, 
                      top_k=50):
        """
        Generate text from a given prompt
        
        Args:
            prompt (str): Starting text for generation
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k filtering for token selection
        
        Returns:
            str: Generated text
        """
        try:
            # Tokenize the prompt
            tokenized_prompt = self.tokenizer.tokenize(prompt)
            
            # Convert to tensor
            x = torch.tensor(tokenized_prompt, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Generate tokens
            y = self.model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
            
            # Decode tokens
            generated_text = self.tokenizer.decode(y.squeeze().tolist())
            logger.info("Text generated successfully")
            return generated_text
        
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""

def main():
    """Main text generation script"""
    try:
        generator = TextGenerator(model_path='./model/model.pth')
        
        # Example prompts for generation
        prompt = str(input("Enter a prompt: "))
        
        print(f"\nGenerating text from prompt: {prompt}\n")
        generated_text = generator.generate_text(prompt)
        print(generated_text)
    
    except Exception as e:
        logger.error(f"Generation script failed: {e}")

if __name__ == '__main__':
    main()