import numpy as np
import pytest
from cleanlab.lexical_quality import (
    calculate_accuracy,
    calculate_grammar_quality,
    calculate_readability,
    calculate_lexical_quality,
    compute_quality_metrics
)

def test_calculate_accuracy():
    assert calculate_accuracy("this is a test.") == 1.0  
    assert calculate_accuracy("Ths is a tst.") < 1.0  
    assert calculate_accuracy("") == 1.0  

def test_calculate_grammar_quality():
    assert calculate_grammar_quality("This is a correct sentence.") == 1.0  
    assert calculate_grammar_quality("This are incorrect.") < 1.0  
    assert calculate_grammar_quality("") == 1.0  

def test_calculate_readability():
    assert calculate_readability("This is a simple sentence.") >= 0.0  
    assert calculate_readability("The quintessential quintessence of quintessentially quintessential qualities is quintessentially quintessential") == 0.0  

def test_calculate_lexical_quality():
    text = "This is a test sentence."
    quality_score = calculate_lexical_quality(text)
    assert 0.0 <= quality_score <= 1.0  

def test_compute_quality_metrics():
    texts = np.array([
        "This is a test.",
        "Ths is a tst.",
        "This are incorrect."
    ])
    metrics = compute_quality_metrics(texts)
    assert metrics.shape == (3,)  
    assert np.all((0.0 <= metrics) & (metrics <= 1.0))  