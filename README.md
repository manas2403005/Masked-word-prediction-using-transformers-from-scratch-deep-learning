# Masked-word-prediction-using-transformers-from-scratch-deep-learning


This project implements a masked word prediction model entirely from scratch using a Transformer-based architecture. Starting with LSTM and BiLSTM models, we transitioned to Transformers for improved performance. Evaluation was done using both semantic similarity and sentence fluency metrics, all without relying on pre-trained models for prediction.

---

## Project Highlights

* Built from scratch — no pre-trained models used for prediction
* Iterative development: LSTM → BiLSTM → Transformer
* Custom evaluation using **Cosine Similarity** and **Perplexity**
* Trained on constrained hardware (Google Colab, Kaggle)
* Achieved strong contextual understanding and fluency

---

## Model Development Timeline

**1. LSTM Baseline:**
Initial experiments with LSTM models yielded poor results. Training was slow, and the model failed to capture long-range context.

**2. BiLSTM Enhancement:**
While BiLSTM slightly improved performance by capturing bidirectional context, it was still inefficient and struggled with complex sentence structures.

**3. Transition to Transformers:**
Implemented a Transformer model to leverage self-attention mechanisms. Key improvements included:

* Better context handling across sentences
* Higher accuracy and semantic alignment
* Required optimization due to increased training time

**4. Training and Computational Challenges:**
Faced limited GPU access and memory constraints. Addressed with:

* Precision reduction (`torch.float16`)
* Batch splitting
* Manual resource management (session restarts, parallelization)

**5. Hyperparameter Tuning:**
Optimized parameters such as:

* Optimizer: Switched to AdamW for stability
* Embedding dimensions, number of attention heads
* Batch size and learning rate
* Epoch count for efficient convergence

---

## Evaluation Metrics

To assess the model's performance, we used two key metrics:

**1. Cosine Similarity (using Sentence-BERT):**
Measures the semantic similarity between our model's predicted word and BERT's prediction.

* Allows for flexible, context-aware comparisons
* Average similarity score: **54.7**
* Limitation: May undervalue valid but diverse predictions

**2. Perplexity (using GPT-2):**
Evaluates how fluent the full sentence is after word insertion.

* Lower score indicates more natural, grammatically correct sentences
* Average perplexity: **488**
* Limitation: Favors common word usage; doesn't directly evaluate correctness

---

## Final Outputs

* Model predictions and their evaluation metrics are saved in structured CSV files
* Merged results allow for comprehensive analysis and comparison
* Outputs are formatted for easy visualization and further benchmarking

---

## Conclusion

This project demonstrates that with careful architecture design and optimization, a Transformer model built from scratch can effectively predict masked words. Our approach outperformed LSTM and BiLSTM baselines in both semantic alignment and sentence fluency. The use of multiple evaluation metrics ensures a more balanced and reliable assessment of performance.

---


