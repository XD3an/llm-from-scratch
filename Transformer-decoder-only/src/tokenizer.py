import tiktoken


class TextTokenizer:
    def __init__(self, encoding_name="cl100k_base"):
        """
        Initialize tokenizer with optional encoding selection
        
        Args:
            encoding_name (str): Tiktoken encoding name
        """
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            raise

    def tokenize(self, text):
        """
        Tokenize input text
        
        Args:
            text (str): Input text to tokenize
        
        Returns:
            list: List of token ids
        """
        try:
            tokenized_data = self.tokenizer.encode(text)
            return tokenized_data
        except Exception as e:
            print(f"Tokenization error: {e}")
            return []

    def decode(self, token_ids):
        """
        Decode token ids back to text
        
        Args:
            token_ids (list): List of token ids
        
        Returns:
            str: Decoded text
        """
        try:
            return self.tokenizer.decode(token_ids)
        except Exception as e:
            print(f"Decoding error: {e}")
            return ""

def tokenize_data(data):
    """
    Global tokenization function with additional error handling
    
    Args:
        data (str): Text to tokenize
    
    Returns:
        list: Tokenized data
    """
    try:
        tokenizer = TextTokenizer()
        tokenized_data = tokenizer.tokenize(data)
        
        # Optional: Track max token value if needed
        global max_token_value
        max_token_value = max(tokenized_data) if tokenized_data else 0
        
        return tokenized_data
    except Exception as e:
        print(f"Data tokenization failed: {e}")
        return []