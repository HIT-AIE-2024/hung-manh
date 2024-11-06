import nltk
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List

# Ensure required nltk resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

class Tokenize:
    """
    A class for tokenizing and normalizing text.
    
    This class provides methods to tokenize text using different tokenizers
    and to normalize tokens through stemming and lemmatization.
    
    Attributes:
    -----------
    text : str
        The input text to be tokenized and normalized.
    """

    def __init__(self, text: str):
        """
        Initializes the Tokenize class with the provided text.
        
        Parameters:
        ----------
        text : str
            The input text to be processed.
        """
        self.text = text

    def whitespace_tokenizer(self) -> List[str]:
        """
        Tokenizes the text based on whitespace.
        
        This method splits the text into tokens by whitespace.
        
        Returns:
        -------
        List[str]
            A list of tokens separated by whitespace.
        """
        tokenizer = WhitespaceTokenizer()
        return tokenizer.tokenize(self.text)

    def word_punct_tokenizer(self) -> List[str]:
        """
        Tokenizes the text using WordPunctTokenizer.
        
        This tokenizer splits the text into words and punctuation marks as separate tokens.
        
        Returns:
        -------
        List[str]
            A list of tokens, with punctuation marks as separate tokens.
        """
        tokenizer = WordPunctTokenizer()
        return tokenizer.tokenize(self.text)

    def treebank_word_tokenizer(self) -> List[str]:
        """
        Tokenizes the text using TreebankWordTokenizer.
        
        This tokenizer splits the text according to rules based on the Penn Treebank corpus.
        It handles punctuation and contractions more effectively for English text.
        
        Returns:
        -------
        List[str]
            A list of tokens based on Treebank tokenization rules.
        """
        tokenizer = TreebankWordTokenizer()
        return tokenizer.tokenize(self.text)

if __name__ == "__main__":
    # Example usage
    text = "Đây là một đoạn văn bản mẫu để thử nghiệm các phương pháp tokenization."
    tokenizer = Tokenize(text)

    # Tokenization
    tokens_whitespace = tokenizer.whitespace_tokenizer()
    tokens_word_punct = tokenizer.word_punct_tokenizer()
    tokens_treebank = tokenizer.treebank_word_tokenizer()

    # print tokens
    print("Tokens (Whitespace):", tokens_whitespace)
    print("Tokens (WordPunct):", tokens_word_punct)
    print("Tokens (Treebank):", tokens_treebank)
