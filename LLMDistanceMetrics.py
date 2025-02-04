import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

class LLMDistanceMetrics:
    def __init__(self):
        # Initialize NLTK resources
        try:
            nltk.download('wordnet')
            nltk.download('punkt')
        except:
            print("Warning: NLTK resources could not be downloaded")
        
        self.rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    def jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two probability distributions
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            float: JSD value between 0 and 1
        """
        # Ensure proper probability distributions
        p = np.asarray(p)
        q = np.asarray(q)
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))

    def compute_bleu(self, reference: str, candidate: str) -> float:
        """
        Compute BLEU score between reference and candidate texts
        
        Args:
            reference: Reference text
            candidate: Candidate text
            
        Returns:
            float: BLEU score between 0 and 1
        """
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        return sentence_bleu([reference_tokens], candidate_tokens)

    def compute_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Compute ROUGE scores between reference and candidate texts
        
        Args:
            reference: Reference text
            candidate: Candidate text
            
        Returns:
            Dict containing ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        scores = self.rouge_calculator.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    @staticmethod
    def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Centered Kernel Alignment between two matrices of activations
        
        Args:
            X: First matrix of shape (n_samples, n_features1)
            Y: Second matrix of shape (n_samples, n_features2)
            
        Returns:
            float: CKA similarity score between 0 and 1
        """
        X = normalize(X)
        Y = normalize(Y)
        
        # Center the matrices
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        
        # Compute kernel matrices
        K_X = X @ X.T
        K_Y = Y @ Y.T
        
        # Compute CKA
        hsic = np.sum(K_X * K_Y)
        norm_X = np.sqrt(np.sum(K_X * K_X))
        norm_Y = np.sqrt(np.sum(K_Y * K_Y))
        
        return hsic / (norm_X * norm_Y)

    def compute_model_agreement(self, outputs1: List[str], outputs2: List[str]) -> float:
        """
        Compute agreement rate between two models' outputs
        
        Args:
            outputs1: List of outputs from first model
            outputs2: List of outputs from second model
            
        Returns:
            float: Agreement rate between 0 and 1
        """
        if len(outputs1) != len(outputs2):
            raise ValueError("Output lists must have same length")
            
        agreements = sum(1 for o1, o2 in zip(outputs1, outputs2) if o1 == o2)
        return agreements / len(outputs1)

    def compute_embedding_distance(self, 
                                model1_name: str, 
                                model2_name: str, 
                                input_texts: List[str]) -> float:
        """
        Compute average cosine distance between embeddings of two models
        
        Args:
            model1_name: Name of first model (from HuggingFace)
            model2_name: Name of second model (from HuggingFace)
            input_texts: List of input texts to compare
            
        Returns:
            float: Average cosine distance between embeddings
        """
        # Load models and tokenizers
        model1 = AutoModel.from_pretrained(model1_name)
        model2 = AutoModel.from_pretrained(model2_name)
        tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
        tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
        
        distances = []
        
        for text in input_texts:
            # Get embeddings from model 1
            inputs1 = tokenizer1(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs1 = model1(**inputs1)
            emb1 = outputs1.last_hidden_state.mean(dim=1).numpy()
            
            # Get embeddings from model 2
            inputs2 = tokenizer2(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs2 = model2(**inputs2)
            emb2 = outputs2.last_hidden_state.mean(dim=1).numpy()
            
            # Compute cosine distance
            distance = cosine(emb1.flatten(), emb2.flatten())
            distances.append(distance)
            
        return np.mean(distances)

    def compute_perplexity(self, 
                         model, 
                         tokenizer, 
                         text: str, 
                         stride: int = 512) -> float:
        """
        Compute perplexity of a text using sliding window approach
        
        Args:
            model: Language model
            tokenizer: Associated tokenizer
            text: Input text
            stride: Stride length for sliding window
            
        Returns:
            float: Perplexity score
        """
        encodings = tokenizer(text, return_tensors="pt")
        max_length = model.config.max_position_embeddings
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            target_len = end_loc - prev_end_loc
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-target_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
                
        return torch.exp(torch.stack(nlls).mean())

def example_usage():
    # Initialize the metrics calculator
    metrics = LLMDistanceMetrics()
    
    # Example texts
    reference = "The quick brown fox jumps over the lazy dog."
    candidate = "A fast brown fox leaps over the sleeping dog."
    
    # Compute BLEU score
    bleu = metrics.compute_bleu(reference, candidate)
    print(f"BLEU score: {bleu}")
    
    # Compute ROUGE scores
    rouge_scores = metrics.compute_rouge(reference, candidate)
    print(f"ROUGE scores: {rouge_scores}")
    
    # Example with embeddings
    embeddings1 = np.random.rand(10, 768)  # Simulated embeddings
    embeddings2 = np.random.rand(10, 768)
    
    # Compute CKA
    cka = metrics.centered_kernel_alignment(embeddings1, embeddings2)
    print(f"CKA similarity: {cka}")
    
    # Example probability distributions
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.4, 0.5])
    
    # Compute JSD
    jsd = metrics.jensen_shannon_divergence(p, q)
    print(f"Jensen-Shannon divergence: {jsd}")

if __name__ == "__main__":
    example_usage()
