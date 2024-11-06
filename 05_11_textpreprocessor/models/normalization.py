import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List

# Ensure that the required nltk resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

class Normalization:
    """
    A class for normalizing text through stemming and lemmatization.
    
    This class provides methods to normalize tokens by reducing them
    to their root form or base meaning.
    """

    def __init__(self, tokens: List[str]):
        """
        Initializes the Normalization class with a list of tokens.
        
        Parameters:
        ----------
        tokens : List[str]
            The list of tokens to be normalized.
        """
        self.tokens = tokens
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def stemming(self) -> List[str]:
        """
        Apply stemming to each token in the list.
        
        Stemming reduces each word to its root form by removing suffixes.
        
        Returns:
        -------
        List[str]
            A list of stemmed tokens.
        """
        return [self.stemmer.stem(token) for token in self.tokens]

    def lemmatization(self) -> List[str]:
        """
        Apply lemmatization to each token in the list.
        
        Lemmatization reduces each word to its base or dictionary form.
        
        Returns:
        -------
        List[str]
            A list of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in self.tokens]


# Example usage
if __name__ == "__main__":
    # Sample tokens
    tokens = ['Đây', 'là', 'một', 'đoạn', 'văn', 'bản', 'mẫu', 'để', 'thử', 'nghiệm', 'các', 'phương', 'pháp', 'tokenization', '.']

    # Create an instance of the Normalization class
    normalizer = Normalization(tokens)

    # Apply stemming
    stemmed_tokens = normalizer.stemming()
    print("Stemmed Tokens:", stemmed_tokens)

    # Apply lemmatization
    lemmatized_tokens = normalizer.lemmatization()
    print("Lemmatized Tokens:", lemmatized_tokens)
