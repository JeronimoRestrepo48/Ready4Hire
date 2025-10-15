# Advanced ML Algorithms for Question Selection
## Ready4Hire - Technical Documentation

**Version**: 2.0  
**Last Updated**: October 2025  
**Author**: Ready4Hire ML Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Algorithms](#algorithms)
4. [Training Pipeline](#training-pipeline)
5. [Continuous Learning](#continuous-learning)
6. [Performance Metrics](#performance-metrics)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Ready4Hire system uses advanced machine learning techniques to intelligently select interview questions based on candidate profiles, interview context, and historical performance data.

### Key Features

- **Semantic Search**: SentenceTransformers for embedding-based similarity
- **Unsupervised Clustering**: UMAP + HDBSCAN for topic discovery
- **Ranking**: RankNet neural network for relevance scoring
- **Continuous Learning**: Online learning from interview feedback
- **Multi-armed Bandits**: Exploration vs exploitation balance

### Tech Stack

- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Dimensionality Reduction**: `umap-learn`
- **Clustering**: `hdbscan`, `scikit-learn`
- **Deep Learning**: `PyTorch`
- **Metrics**: `scikit-learn`, custom implementations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Question Selection Pipeline                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Input: Candidate Profile + Interview Context           │
│     ↓                                                       │
│  2. Embedding Generation (SentenceTransformers)            │
│     ↓                                                       │
│  3. Clustering (UMAP + HDBSCAN)                            │
│     ├─ Topic Discovery                                     │
│     ├─ Question Grouping                                   │
│     └─ Coverage Analysis                                   │
│     ↓                                                       │
│  4. Ranking (RankNet + Continuous Learning)                │
│     ├─ Semantic Similarity                                 │
│     ├─ Historical Performance                              │
│     ├─ Exploration Bonus                                   │
│     └─ Diversity Penalty                                   │
│     ↓                                                       │
│  5. Selection Strategy                                     │
│     ├─ Diversified Sampling                                │
│     ├─ Difficulty Adaptation                               │
│     └─ Theme Balancing                                     │
│     ↓                                                       │
│  6. Output: Optimized Question Sequence                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Embeddings      │─────▶│  Clustering      │─────▶│  Ranking         │
│  Service         │      │  Service         │      │  Service         │
│                  │      │                  │      │                  │
│ • SentenceTrans. │      │ • UMAP           │      │ • RankNet        │
│ • Caching        │      │ • HDBSCAN        │      │ • Performance    │
│ • Batch Process  │      │ • K-Means        │      │ • Multi-Bandit   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
                                                              │
                                                              ▼
                                                     ┌──────────────────┐
                                                     │  Continuous      │
                                                     │  Learning        │
                                                     │                  │
                                                     │ • Feedback Loop  │
                                                     │ • Online Update  │
                                                     │ • A/B Testing    │
                                                     └──────────────────┘
```

---

## Algorithms

### 1. Embedding Generation

**Algorithm**: SentenceTransformers (all-MiniLM-L6-v2)

**Purpose**: Convert text questions to dense vector representations

**Implementation**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)  # Shape: (n_texts, 384)
```

**Properties**:
- Dimension: 384
- Pre-trained on: Semantic similarity tasks
- Supports: Multi-lingual (100+ languages)
- Inference time: ~10ms per sentence (CPU)

**Advantages**:
- ✅ Captures semantic meaning
- ✅ Fast inference
- ✅ Small model size (~80MB)
- ✅ Good for short texts

**Limitations**:
- ⚠️ Fixed context window (256 tokens)
- ⚠️ Domain adaptation may improve results

---

### 2. Dimensionality Reduction (UMAP)

**Algorithm**: Uniform Manifold Approximation and Projection

**Purpose**: Reduce embedding dimensions while preserving structure

**Mathematical Foundation**:
```
UMAP optimizes:
  min Σ (1 + a*d^(2b)) * (1 - y_ij) * log(1 - y_ij) + y_ij * log(y_ij)

Where:
  - d: distance in high-dimensional space
  - y: membership strength in low-dimensional space
  - a, b: hyperparameters
```

**Implementation**:
```python
import umap

reducer = umap.UMAP(
    n_components=10,        # Target dimensions
    n_neighbors=15,         # Local neighborhood size
    min_dist=0.1,          # Minimum distance between points
    metric='cosine',       # Distance metric
    random_state=42
)

reduced_embeddings = reducer.fit_transform(embeddings)
```

**Hyperparameters**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_components` | 10 | Balance between information retention and clustering |
| `n_neighbors` | 15 | Capture local structure without over-smoothing |
| `min_dist` | 0.1 | Allow tight clusters while preventing overlap |
| `metric` | cosine | Best for semantic embeddings |

**Performance**:
- Time complexity: O(n log n)
- Space complexity: O(n²) for distance matrix
- Typical runtime: ~2s for 1000 samples (CPU)

---

### 3. Clustering (HDBSCAN)

**Algorithm**: Hierarchical Density-Based Spatial Clustering of Applications with Noise

**Purpose**: Group similar questions into thematic clusters

**Key Concepts**:
1. **Core Distance**: Minimum distance to k-th nearest neighbor
2. **Mutual Reachability**: max(core_dist(a), core_dist(b), dist(a,b))
3. **Minimum Spanning Tree**: Connect all points with minimum total distance
4. **Cluster Hierarchy**: Extract stable clusters from dendrogram
5. **Stability**: Measure cluster persistence across scales

**Implementation**:
```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,            # Minimum cluster size
    min_samples=3,                 # Core point threshold
    metric='euclidean',            # Distance metric
    cluster_selection_method='eom' # Excess of Mass
)

cluster_labels = clusterer.fit_predict(reduced_embeddings)
```

**Advantages over K-Means**:
- ✅ **No need to specify K**: Automatically finds optimal number of clusters
- ✅ **Handles noise**: Points that don't fit are labeled as -1 (noise)
- ✅ **Variable density**: Works with clusters of different densities
- ✅ **Hierarchical structure**: Preserves cluster relationships
- ✅ **Outlier detection**: Built-in anomaly detection

**Cluster Quality Metrics**:
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette Score: Higher is better [-1, 1]
silhouette = silhouette_score(embeddings, labels)

# Davies-Bouldin Index: Lower is better [0, ∞)
db_index = davies_bouldin_score(embeddings, labels)
```

**Typical Results**:
- Number of clusters: 8-15 (for 100-500 questions)
- Noise points: 5-15% of total
- Silhouette score: 0.3-0.5 (good for text data)

---

### 4. Ranking (RankNet)

**Algorithm**: Pairwise Ranking Neural Network

**Purpose**: Score questions based on relevance to context

**Architecture**:
```
Input (context ⊕ question embeddings): 768 dimensions
  ↓
Dense Layer 1: 768 → 64 (ReLU)
  ↓
Dense Layer 2: 64 → 32 (ReLU)
  ↓
Output Layer: 32 → 1 (score)
```

**Training Objective**:
```
RankNet Loss:
  L = -Σ (P_ij * log(σ(s_i - s_j)))

Where:
  - P_ij = 1 if question i > question j (preference)
  - s_i, s_j = scores from network
  - σ = sigmoid function
```

**Implementation**:
```python
import torch
import torch.nn as nn

class RankNet(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**Training Data Format**:
```python
# Pairs of (better_question, worse_question) based on:
# - Historical performance
# - User feedback
# - Expert annotations

training_pairs = [
    (emb_q1, emb_q2, label=1),  # q1 > q2
    (emb_q3, emb_q4, label=0),  # q3 < q4
    ...
]
```

**Hyperparameters**:
- Learning rate: 0.001
- Batch size: 32
- Epochs: 50-100
- Optimizer: Adam
- Regularization: Dropout(0.2)

---

### 5. Continuous Learning

**Algorithm**: Online Learning with Multi-Armed Bandits

**Purpose**: Improve question selection over time without retraining

**Techniques**:

#### 5.1 Epsilon-Greedy Exploration
```python
if random() < epsilon:
    # Explore: Select random question
    question = random.choice(available_questions)
else:
    # Exploit: Select best known question
    question = max(available_questions, key=lambda q: q.avg_performance)
```

**Epsilon Schedule**:
- Initial: ε = 0.3 (high exploration)
- Decay: ε = ε * 0.995 per iteration
- Minimum: ε = 0.05 (always explore 5%)

#### 5.2 Upper Confidence Bound (UCB)
```python
def ucb_score(question, t):
    """
    UCB1 Algorithm
    
    score = avg_reward + sqrt(2 * ln(t) / n_i)
    
    Where:
      - avg_reward: average performance of question
      - t: total number of selections
      - n_i: times this question was selected
    """
    exploration_bonus = np.sqrt(2 * np.log(t) / question.times_asked)
    return question.avg_performance + exploration_bonus
```

#### 5.3 Thompson Sampling
```python
def thompson_sample(question):
    """
    Sample from posterior distribution of question quality
    
    Assumes Beta distribution: Beta(α, β)
      - α = successes + 1
      - β = failures + 1
    """
    alpha = question.good_responses + 1
    beta = question.bad_responses + 1
    return np.random.beta(alpha, beta)
```

**Performance Metrics**:
```python
@dataclass
class QuestionPerformance:
    question_id: str
    times_asked: int
    avg_score: float                 # Mean response quality
    avg_response_time: float         # Mean time to answer
    difficulty_actual: float         # Empirical difficulty
    discrimination_power: float      # How well it separates candidates
    last_updated: datetime
```

**Update Rules**:
```python
# Exponential Moving Average (EMA)
α = 0.1  # Learning rate

perf.avg_score = (1 - α) * perf.avg_score + α * new_score
perf.avg_time = (1 - α) * perf.avg_time + α * new_time
```

---

## Training Pipeline

### Phase 1: Data Preparation

**Input Data**:
```
app/datasets/
├── tech_questions.jsonl      # Technical questions
├── soft_skills.jsonl         # Soft skills questions
└── finetune_interactions.jsonl  # Historical interviews
```

**Data Format**:
```json
{
  "id": "q_001",
  "question": "Explain the SOLID principles",
  "category": "software_engineering",
  "difficulty": "intermediate",
  "expected_keywords": ["single_responsibility", "open_closed", ...],
  "role": "backend_developer"
}
```

**Preprocessing Steps**:
1. Load questions from JSONL files
2. Clean text (remove special characters, normalize)
3. Validate required fields
4. Split into train/validation sets (80/20)

### Phase 2: Embedding Training

```bash
# Generate embeddings for all questions
python app/infrastructure/ml/train_embeddings.py \
    --input datasets/tech_questions.jsonl \
    --output models/question_embeddings.npy \
    --model all-MiniLM-L6-v2
```

**Output**:
- `question_embeddings.npy`: Numpy array (n_questions × 384)
- `question_metadata.json`: Question IDs and metadata

### Phase 3: Clustering Training

```bash
# Perform clustering
python app/infrastructure/ml/train_clustering.py \
    --embeddings models/question_embeddings.npy \
    --min_cluster_size 5 \
    --output models/clusters.json
```

**Output**:
```json
{
  "clusters": {
    "0": {
      "questions": ["q_001", "q_045", ...],
      "topic": "Object-Oriented Programming",
      "keywords": ["class", "inheritance", "polymorphism"],
      "size": 23
    },
    ...
  },
  "metadata": {
    "n_clusters": 12,
    "silhouette_score": 0.42,
    "algorithm": "hdbscan"
  }
}
```

### Phase 4: RankNet Training

```bash
# Train ranking model
python app/infrastructure/ml/train_ranknet.py \
    --embeddings models/question_embeddings.npy \
    --feedback datasets/finetune_interactions.jsonl \
    --output models/ranknet_model.pt \
    --epochs 100
```

**Training Configuration**:
```python
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'hidden_dims': [64, 32],
    'dropout': 0.2,
    'optimizer': 'adam',
    'loss': 'ranknet_loss'
}
```

**Training Logs**:
```
Epoch 1/100: loss=0.623, val_loss=0.591
Epoch 10/100: loss=0.412, val_loss=0.438
Epoch 50/100: loss=0.298, val_loss=0.321
Epoch 100/100: loss=0.256, val_loss=0.289
✅ Training complete! Model saved to ranknet_model.pt
```

### Phase 5: Evaluation

```bash
# Evaluate models
python app/infrastructure/ml/evaluate.py \
    --model models/ranknet_model.pt \
    --test_data datasets/test_interviews.jsonl
```

**Evaluation Metrics**:
- **NDCG@10** (Normalized Discounted Cumulative Gain): Measures ranking quality
- **MRR** (Mean Reciprocal Rank): Position of first relevant question
- **Precision@K**: Fraction of relevant questions in top K
- **Diversity Score**: Coverage of different clusters
- **User Satisfaction**: Average interview rating

---

## Continuous Learning

### Feedback Collection

```python
# After each interview
feedback = InterviewFeedback(
    interview_id="int_123",
    user_id="user_456",
    role="backend_developer",
    questions_asked=["q_001", "q_023", ...],
    scores=[8.5, 7.2, 9.1, ...],
    response_times=[45.2, 62.1, 38.5, ...],
    final_evaluation=8.3,
    timestamp="2025-10-15T10:30:00Z",
    metadata={"difficulty_level": "intermediate"}
)

learning_system.record_interview_feedback(feedback)
```

### Model Updates

**Incremental Update Schedule**:
- **Real-time**: Update performance metrics after each interview
- **Daily**: Recompute question rankings
- **Weekly**: Retrain RankNet with new feedback
- **Monthly**: Rebuild clusters if significant drift detected

**Drift Detection**:
```python
def detect_drift(current_performance, historical_baseline):
    """
    Kolmogorov-Smirnov test for distribution shift
    """
    from scipy.stats import ks_2samp
    
    statistic, p_value = ks_2samp(current_performance, historical_baseline)
    
    if p_value < 0.05:
        logger.warning("⚠️ Performance drift detected! Consider retraining.")
        return True
    return False
```

### A/B Testing

```python
# Test new selection strategy
strategies = {
    'A': 'exploit',      # Control: Current best
    'B': 'balanced',     # Treatment: New algorithm
}

# Randomly assign users
strategy = random.choice(['A', 'B'])

# Track metrics per strategy
metrics[strategy]['satisfaction'].append(interview_rating)
metrics[strategy]['completion_rate'].append(completed)

# Statistical test (after N samples)
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(metrics['A']['satisfaction'], 
                              metrics['B']['satisfaction'])

if p_value < 0.05 and mean(metrics['B']) > mean(metrics['A']):
    logger.info("✅ Strategy B is significantly better! Deploying...")
```

---

## Performance Metrics

### System Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Embedding Time (per question) | < 20ms | 12ms | ✅ |
| Clustering Time (1000 questions) | < 5s | 3.2s | ✅ |
| Ranking Time (100 candidates) | < 50ms | 35ms | ✅ |
| End-to-End Selection | < 200ms | 150ms | ✅ |
| Memory Usage | < 500MB | 320MB | ✅ |

### Quality Metrics

| Metric | Formula | Target | Current |
|--------|---------|--------|---------|
| NDCG@10 | $\frac{DCG@10}{IDCG@10}$ | > 0.7 | 0.74 |
| MRR | $\frac{1}{N}\sum \frac{1}{rank_i}$ | > 0.6 | 0.68 |
| Diversity | Clusters covered / Total clusters | > 0.8 | 0.85 |
| User Satisfaction | Avg(ratings) | > 4.0/5.0 | 4.2/5.0 |

---

## Usage Examples

### Example 1: Basic Question Selection

```python
from app.infrastructure.ml import get_embeddings_service, get_clustering_service

# Initialize services
embeddings_svc = get_embeddings_service()
clustering_svc = get_clustering_service(embeddings_svc)

# Load questions
questions = load_questions('datasets/tech_questions.jsonl')

# Cluster questions
clusters = clustering_svc.cluster_questions(questions)

# Select diversified questions
selected = clustering_svc.select_diversified_questions(
    n_questions=10,
    candidate_ids=set(q['id'] for q in questions),
    exclude_ids=set(),  # No exclusions
    user_profile={'role': 'backend_developer', 'level': 'intermediate'}
)
```

### Example 2: With Continuous Learning

```python
from app.infrastructure.ml import get_continuous_learning_system

# Initialize learning system
learning_sys = get_continuous_learning_system()

# Get rankings based on performance
rankings = learning_sys.get_question_rankings(
    question_ids=[q['id'] for q in questions],
    strategy='adaptive'  # Adapts to current context
)

# Select top questions
top_questions = [qid for qid, score in rankings[:10]]
```

### Example 3: Analysis Dashboard

```python
# Analyze question pool
analysis = learning_sys.analyze_question_pool()

print(f"Total questions: {analysis['total_questions']}")
print(f"Total interviews: {analysis['total_interviews']}")
print(f"\nTop performing questions:")
for qid, score in analysis['top_performing_questions'][:5]:
    print(f"  {qid}: {score:.2f}/10")

print(f"\nRecommendations:")
for rec in analysis['recommendations']:
    print(f"  {rec}")
```

---

## Troubleshooting

### Issue: Clustering produces only 1-2 clusters

**Possible Causes**:
- `min_cluster_size` too large
- Questions too similar (low diversity)
- UMAP over-smoothed embeddings

**Solutions**:
```python
# Reduce min_cluster_size
clustering_svc = get_clustering_service(
    embeddings_svc,
    min_cluster_size=3  # Instead of 5
)

# Adjust UMAP parameters
umap_reducer = umap.UMAP(
    n_neighbors=10,     # Smaller neighborhood
    min_dist=0.0,       # Allow tighter clusters
    metric='cosine'
)
```

### Issue: Low ranking accuracy

**Possible Causes**:
- Insufficient training data
- Poor quality feedback
- Model overfitting

**Solutions**:
1. Collect more feedback data (target: 1000+ interviews)
2. Implement data validation for feedback quality
3. Add regularization:
```python
model = RankNet(input_dim=768)
# Add dropout
model = nn.Sequential(
    nn.Linear(768, 64),
    nn.ReLU(),
    nn.Dropout(0.3),  # Increased dropout
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 1)
)
```

### Issue: Slow inference time

**Optimization Strategies**:
1. **Batch Processing**:
```python
# Process multiple questions at once
embeddings = model.encode(questions, batch_size=32)
```

2. **Caching**:
```python
# Cache embeddings
@lru_cache(maxsize=1000)
def get_embedding(question_text):
    return model.encode([question_text])[0]
```

3. **Model Quantization**:
```python
# Convert to ONNX for faster inference
torch.onnx.export(model, dummy_input, "ranknet.onnx")
```

---

## References

1. **SentenceTransformers**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.

2. **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.

3. **HDBSCAN**: Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates.

4. **RankNet**: Burges, C., et al. (2005). Learning to Rank using Gradient Descent.

5. **Multi-Armed Bandits**: Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.

---

**For more information, see:**
- API Documentation: `/docs`
- Code Examples: `/examples`
- GitHub Issues: [Report bugs](https://github.com/Ready4Hire/issues)
