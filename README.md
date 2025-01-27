# Deep Learning Project - Image Caption Generator

# Image Captioning Model Architecture

## Model Implementation Details
The model is implemented in PyTorch as a neural network that combines image features with text sequences to generate captions.

## Input Processing
### Image Input
- **Input Shape**: (batch_size, 2048)
  - Pre-extracted features from a feature extractor
  - Replaces the original convolutional layers for efficiency

### Text Input
- **Input Shape**: (batch_size, sequence_length)
- **Type**: Long tensor of token indices

## Model Components

### Image Feature Processing
```python
# Image feature dense layer
fc1 = nn.Linear(2048, 256)
dropout1 = nn.Dropout(0.2)
```
- Takes 2048-dimensional image features
- Projects to 256-dimensional space
- Applies ReLU activation
- 20% dropout for regularization

### Text Sequence Processing
```python
# Embedding layer
embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
dropout2 = nn.Dropout(0.2)
lstm = nn.LSTM(embedding_dim, 256, batch_first=True)
```
- Embeds tokens into continuous vector space
- Embedding dimension is configurable
- Handles padding with padding_idx=0
- 20% dropout after embedding
- Single-layer LSTM with 256 hidden units
- Uses batch_first=True for easier batch processing

### Feature Fusion and Decoder
```python
# Decoder layers
fc2 = nn.Linear(256 + 256, 256)  # Combines image and text features
fc3 = nn.Linear(256, vocabulary_size)  # Output layer
```
1. Feature Combination
   - Concatenates 256-dim image features with 256-dim LSTM output
   - Total concatenated dimension: 512 (256 + 256)

2. Decoder Network
   - First dense layer: 512 → 256 with ReLU
   - Final output layer: 256 → vocabulary_size
   - Softmax applied implicitly through loss function

## Forward Pass Flow
1. Image features pass through first dense layer (fc1)
2. Text tokens are embedded and processed by LSTM
3. Last LSTM output is extracted
4. Features are concatenated
5. Decoder processes combined features
6. Final layer produces vocabulary-sized logits

## Training Details
- **Device**: Supports both CPU and GPU training
- **Loss Function**: Typically CrossEntropyLoss (not shown in model definition)
- **Regularization**: Two dropout layers (20% each)
- **Batch Processing**: Fully vectorized operations for efficient training

## Architecture Notes
- The model uses pre-extracted image features (2048-dim) instead of raw images
- This is more efficient than the previous convolutional approach
- LSTM processes entire sequences but only uses final output
- The architecture is memory-efficient due to:
  - Pre-extracted image features
  - Single-layer LSTM
  - Moderate hidden dimensions (256)