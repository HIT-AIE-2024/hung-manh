from models.tokenization import Tokenize
from models.normalization import Normalization

    
if __name__ == "__main__":
    # Example usage
    text = "Đây là một đoạn văn bản mẫu để thử nghiệm các phương pháp tokenization và normalization."

    tokenizer = Tokenize(text)

    # Tokenization
    tokens_whitespace = tokenizer.whitespace_tokenizer()
    tokens_word_punct = tokenizer.word_punct_tokenizer()
    tokens_treebank = tokenizer.treebank_word_tokenizer()

    # print tokens
    print("Tokens (Whitespace):", tokens_whitespace)
    print("Tokens (WordPunct):", tokens_word_punct)
    print("Tokens (Treebank):", tokens_treebank)
    
    normalizer = Normalization(tokens_treebank)
    # Apply stemming
    stemmed_tokens = normalizer.stemming()
    print("Stemmed Tokens:", stemmed_tokens)

    # Apply lemmatization
    lemmatized_tokens = normalizer.lemmatization()
    print("Lemmatized Tokens:", lemmatized_tokens)