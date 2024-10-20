from textblob import TextBlob
import language_tool_python
import textstat
import numpy as np

tool = language_tool_python.LanguageTool('en-US')

def calculate_accuracy(text: str) -> float:
    """
    Calculates the spelling accuracy of the given text by comparing 
    the original text to the spell-corrected version using TextBlob.

    Args:
        text (str): Input text to evaluate.

    Returns:
        float: A value between 0.0 and 1.0, where 1.0 indicates perfect spelling.
    """
    words = text.split()
    corrected_words = TextBlob(text).correct().split()
    mismatches = sum(w1 != w2 for w1, w2 in zip(words, corrected_words))
    return 1 - (mismatches / len(words)) if words else 1.0


def calculate_grammar_quality(text: str) -> float:
    """
    Evaluates the grammatical correctness of the input text using 
    LanguageTool, returning a score between 0.0 and 1.0.

    Args:
        text (str): Input text to evaluate.

    Returns:
        float: A score where 1.0 means no grammar issues and 0.0 indicates many issues.
    """
    errors = len(tool.check(text))
    words_count = len(text.split())
    return max(1 - (errors / words_count), 0.0) if words_count else 1.0


def calculate_readability(text: str) -> float:
    """
    Calculates the readability of the input text based on the 
    Flesch Reading Ease score and scales it between 0.0 and 1.0.

    Args:
        text (str): Input text to evaluate.

    Returns:
        float: A readability score between 0.0 and 1.0.
    """
    readability_score = textstat.flesch_reading_ease(text)
    return max(min(readability_score / 100, 1.0), 0.0)


def calculate_lexical_quality(text: str) -> float:
    """
    Computes the overall lexical quality of the text by averaging 
    the spelling, grammar, and readability scores.

    Args:
        text (str): Input text to evaluate.

    Returns:
        float: A lexical quality score between 0.0 and 1.0.
    """
    accuracy = calculate_accuracy(text)
    grammar = calculate_grammar_quality(text)
    readability = calculate_readability(text)
    return (accuracy + grammar + readability) / 3


def compute_quality_metrics(X_raw: np.ndarray) -> np.ndarray:
    """
    Compute lexical quality metrics for an array of texts.

    Parameters:
        X_raw (np.ndarray): A NumPy array containing text data.

    Returns:
        np.ndarray: A NumPy array of lexical quality metrics for each text in the array.
    """

    # Ensure the input is a NumPy array
    if not isinstance(X_raw, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Apply the calculation function to each text in the array using vectorization
    metrics = np.vectorize(calculate_lexical_quality)(X_raw)

    return metrics
