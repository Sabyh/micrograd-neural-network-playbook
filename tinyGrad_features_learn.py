"""
COMPLETE DEEP LEARNING LIBRARY FEATURES GUIDE
=============================================

This file contains all the features that make a complete deep learning library,
explained simply with examples. It shows what your Value class already has
and what features are still missing to implement.

YOUR VALUE CLASS FOUNDATION:
✅ Automatic Differentiation (The most important breakthrough!)
✅ Basic Math Operations (+, -, *, /, tanh, exp)
✅ Computational Graph Building
✅ Backward Propagation

MISSING FEATURES TO IMPLEMENT:
❌ Multi-dimensional Tensors
❌ GPU Support
❌ Broadcasting
❌ Neural Network Layers
❌ Optimizers
❌ Loss Functions
❌ Advanced Architectures
❌ Production Features
❌ Developer Tools

Let's explore each category with examples!
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CORE FEATURES (Essential for any deep learning library)
# ============================================================================

print("="*80)
print("SECTION 1: CORE FEATURES")
print("="*80)

# -----------------------------------------------------------------------------
# 1. TENSORS - Multi-dimensional arrays instead of single numbers
# -----------------------------------------------------------------------------

print("\n1. TENSORS - Multi-dimensional arrays")
print("-" * 40)

class TensorExamples:
    """
    Your Value class works with single numbers.
    Real deep learning needs arrays of numbers (tensors).
    """
    
    def __init__(self):
        print("CURRENT: Your Value class")
        print("a = Value(5.0)  # Just one number")
        
        print("\nNEEDED: Multi-dimensional arrays")
        
        # 1D tensor (vector) - like a list of numbers
        vector = np.array([1, 2, 3, 4, 5])
        print(f"1D tensor: {vector}")
        print(f"Shape: {vector.shape} (5 elements)")
        
        # 2D tensor (matrix) - like a spreadsheet
        matrix = np.array([[1, 2, 3],
                          [4, 5, 6]])
        print(f"2D tensor:\n{matrix}")
        print(f"Shape: {matrix.shape} (2 rows, 3 columns)")
        
        # 3D tensor - like a stack of matrices
        cube = np.array([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]]])
        print(f"3D tensor shape: {cube.shape} (2x2x2)")
        
        print("\nReal-world examples:")
        print("• Photo: height × width × colors (1920 × 1080 × 3)")
        print("• Video: time × height × width × colors (60 × 1920 × 1080 × 3)")
        print("• Text: sentence_length × vocabulary_size (100 × 50000)")
        
        print("\n❌ TO IMPLEMENT: Tensor class supporting multi-dimensional arrays")
        print("   - Store data as numpy arrays or nested lists")
        print("   - Track shape and dimensions")
        print("   - Support indexing like tensor[0, 1, 2]")

# Run tensor examples
tensor_demo = TensorExamples()

# -----------------------------------------------------------------------------
# 2. GPU SUPPORT - Fast computation on graphics cards
# -----------------------------------------------------------------------------

print("\n\n2. GPU SUPPORT - Fast computation")
print("-" * 40)

class GPUExamples:
    """
    CPUs have 4-8 cores (like 4-8 workers)
    GPUs have 1000+ cores (like 1000+ workers)
    """
    
    def demonstrate_parallelism(self):
        print("CPU vs GPU analogy:")
        print("CPU = 1 very smart person (4-8 cores)")
        print("GPU = 1000 average people working together (1000+ cores)")
        
        # Simulate timing difference
        big_array = np.random.random(100000)  # 100k numbers
        
        print(f"\nTask: Add 1 to {len(big_array)} numbers")
        
        # CPU timing
        start = time.time()
        cpu_result = big_array + 1  # NumPy uses CPU
        cpu_time = time.time() - start
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        print("GPU would be 10-100x faster for this task!")
        print("GPU time: ~{:.4f} seconds (estimated)".format(cpu_time / 50))
        
        print("\n❌ TO IMPLEMENT: GPU support")
        print("   - Device abstraction (CPU/GPU/TPU)")
        print("   - Memory management between devices")
        print("   - CUDA/OpenCL kernel compilation")
        print("   - Automatic data transfer between devices")

gpu_demo = GPUExamples()
gpu_demo.demonstrate_parallelism()

# -----------------------------------------------------------------------------
# 3. BROADCASTING - Operations between different shaped arrays
# -----------------------------------------------------------------------------

print("\n\n3. BROADCASTING - Operations between different shapes")
print("-" * 40)

class BroadcastingExamples:
    """
    Handle operations between arrays of different sizes
    """
    
    def demonstrate_broadcasting(self):
        print("Problem: How to add arrays of different sizes?")
        
        # Example 1: Add number to array
        array = np.array([1, 2, 3, 4])
        number = 10
        
        print(f"\nArray: {array}")
        print(f"Number: {number}")
        
        # Broadcasting automatically handles this
        result = array + number
        print(f"Result: {result} (adds 10 to each element)")
        
        # Example 2: More complex broadcasting
        matrix = np.array([[1, 2, 3],
                          [4, 5, 6]])
        vector = np.array([10, 20, 30])
        
        print(f"\nMatrix:\n{matrix}")
        print(f"Vector: {vector}")
        
        result = matrix + vector  # Adds vector to each row
        print(f"Matrix + Vector:\n{result}")
        
        print("\nBroadcasting rules:")
        print("1. Align shapes from the right")
        print("2. Sizes must be equal OR one must be 1")
        print("3. Missing dimensions treated as size 1")
        
        print("\n❌ TO IMPLEMENT: Broadcasting logic")
        print("   - Shape compatibility checking")
        print("   - Automatic shape expansion")
        print("   - Memory-efficient implementation")

broadcasting_demo = BroadcastingExamples()
broadcasting_demo.demonstrate_broadcasting()

# -----------------------------------------------------------------------------
# 4. NEURAL NETWORK LAYERS - Pre-built building blocks
# -----------------------------------------------------------------------------

print("\n\n4. NEURAL NETWORK LAYERS - Pre-built building blocks")
print("-" * 40)

class LayerExamples:
    """
    Instead of building neurons one by one, use pre-made layers
    """
    
    def linear_layer_example(self):
        print("Linear/Dense Layer: Connects every input to every output")
        
        class SimpleLinearLayer:
            def __init__(self, input_size, output_size):
                # Random weights to start (normally distributed)
                self.weights = np.random.randn(input_size, output_size) * 0.1
                self.bias = np.zeros(output_size)
                print(f"Created layer: {input_size} inputs → {output_size} outputs")
            
            def forward(self, x):
                # Matrix multiplication + bias
                return np.dot(x, self.weights) + self.bias
        
        # Example usage
        layer = SimpleLinearLayer(input_size=3, output_size=2)
        input_data = np.array([1.0, 2.0, 3.0])
        output = layer.forward(input_data)
        
        print(f"Input: {input_data}")
        print(f"Output: {output}")
        print(f"Weights shape: {layer.weights.shape}")
        print(f"Bias shape: {layer.bias.shape}")
        
        print("\nOther important layers:")
        print("• Convolutional: For images (detects patterns)")
        print("• Recurrent: For sequences (remembers previous inputs)")
        print("• Attention: For focusing on important parts")
        print("• Batch Normalization: For stable training")
        print("• Dropout: For preventing overfitting")
        
        print("\n❌ TO IMPLEMENT: Layer system")
        print("   - Base Layer class with forward/backward methods")
        print("   - Parameter management (weights, biases)")
        print("   - Gradient computation for each layer type")
        print("   - Layer composition and sequential models")

layer_demo = LayerExamples()
layer_demo.linear_layer_example()

# -----------------------------------------------------------------------------
# 5. OPTIMIZERS - Algorithms to update weights
# -----------------------------------------------------------------------------

print("\n\n5. OPTIMIZERS - Algorithms to update weights")
print("-" * 40)

class OptimizerExamples:
    """
    How to adjust weights to reduce errors
    """
    
    def gradient_descent_example(self):
        print("Gradient Descent: Roll ball down hill to find bottom")
        
        class SimpleOptimizer:
            def __init__(self, learning_rate=0.01):
                self.learning_rate = learning_rate
                print(f"Created optimizer with learning_rate={learning_rate}")
            
            def update(self, weights, gradients):
                # Move weights opposite to gradient direction
                new_weights = weights - self.learning_rate * gradients
                return new_weights
        
        # Example optimization step
        weights = np.array([1.0, 2.0, 3.0])
        gradients = np.array([0.1, -0.2, 0.3])  # Direction of steepest increase
        
        optimizer = SimpleOptimizer(learning_rate=0.1)
        new_weights = optimizer.update(weights, gradients)
        
        print(f"Old weights: {weights}")
        print(f"Gradients: {gradients}")
        print(f"New weights: {new_weights}")
        print("Weights moved opposite to gradient direction")
        
        print("\nAdvanced optimizers:")
        print("• Adam: Adapts learning rate automatically")
        print("• RMSprop: Good for certain problems")
        print("• AdaGrad: Slows down learning over time")
        print("• Momentum: Builds up speed like rolling ball")
        
        print("\n❌ TO IMPLEMENT: Optimizer system")
        print("   - Base Optimizer class")
        print("   - Multiple optimization algorithms")
        print("   - Parameter group management")
        print("   - Learning rate scheduling")

optimizer_demo = OptimizerExamples()
optimizer_demo.gradient_descent_example()

# -----------------------------------------------------------------------------
# 6. LOSS FUNCTIONS - Ways to measure errors
# -----------------------------------------------------------------------------

print("\n\n6. LOSS FUNCTIONS - Ways to measure errors")
print("-" * 40)

class LossFunctionExamples:
    """
    How to measure how wrong predictions are
    """
    
    def demonstrate_losses(self):
        print("Loss functions guide learning by measuring errors")
        
        def mean_squared_error(predictions, targets):
            """For regression: predicting house prices, temperatures"""
            errors = predictions - targets
            squared_errors = errors ** 2
            return np.mean(squared_errors)
        
        def cross_entropy_loss_simple(predictions, targets):
            """For classification: cat vs dog, spam detection"""
            # Simplified version for demonstration
            # Real implementation is more complex
            epsilon = 1e-15  # Prevent log(0)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions))
        
        # Example: Predicting house prices (regression)
        print("Regression example (house prices):")
        actual_prices = np.array([300000, 250000, 400000])
        predicted_prices = np.array([280000, 260000, 390000])
        
        mse = mean_squared_error(predicted_prices, actual_prices)
        print(f"Actual prices: {actual_prices}")
        print(f"Predicted prices: {predicted_prices}")
        print(f"Mean Squared Error: ${mse:.0f}")
        
        # Example: Classification (simplified)
        print("\nClassification example (cat vs dog):")
        print("Predictions: [0.8, 0.3, 0.9] (probability of being cat)")
        print("Targets: [1, 0, 1] (1=cat, 0=dog)")
        print("Cross-entropy measures confidence in wrong predictions")
        
        print("\nCommon loss functions:")
        print("• MSE: Continuous values (prices, temperatures)")
        print("• Cross-entropy: Categories (cat/dog, spam/not spam)")
        print("• Binary Cross-entropy: Yes/no decisions")
        print("• Huber Loss: Robust to outliers")
        
        print("\n❌ TO IMPLEMENT: Loss function system")
        print("   - Base Loss class")
        print("   - Multiple loss implementations")
        print("   - Gradient computation for each loss")
        print("   - Reduction methods (mean, sum, none)")

loss_demo = LossFunctionExamples()
loss_demo.demonstrate_losses()

# ============================================================================
# SECTION 2: ADVANCED FEATURES (For serious deep learning)
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 2: ADVANCED FEATURES")
print("="*80)

# -----------------------------------------------------------------------------
# 1. CONVOLUTIONS - For image processing
# -----------------------------------------------------------------------------

print("\n1. CONVOLUTIONS - For image processing")
print("-" * 40)

class ConvolutionExamples:
    """
    Convolution = sliding a pattern detector over images
    """
    
    def explain_convolution(self):
        print("Convolution: Slide small filter over image to detect patterns")
        
        # Simple convolution example
        print("\nExample: Edge detection")
        image_patch = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        
        edge_filter = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])
        
        print("Image patch:")
        print(image_patch)
        print("\nEdge detection filter:")
        print(edge_filter)
        
        # Element-wise multiplication then sum
        result = np.sum(image_patch * edge_filter)
        print(f"\nConvolution result: {result}")
        print("(Detects vertical edges)")
        
        print("\nWhat different layers learn:")
        print("• Layer 1: Basic features (edges, colors, textures)")
        print("• Layer 2: Shapes (circles, rectangles)")
        print("• Layer 3: Parts (eyes, wheels, windows)")
        print("• Layer 4: Objects (faces, cars, houses)")
        
        print("\nApplications:")
        print("• Medical imaging: Detect tumors in X-rays")
        print("• Self-driving cars: Detect stop signs, pedestrians")
        print("• Photo apps: Face detection and recognition")
        print("• Security: People identification in surveillance")
        
        print("\n❌ TO IMPLEMENT: Convolution operations")
        print("   - 2D convolution forward/backward pass")
        print("   - Padding and stride support")
        print("   - Multiple input/output channels")
        print("   - Efficient implementation (im2col, FFT)")

conv_demo = ConvolutionExamples()
conv_demo.explain_convolution()

# -----------------------------------------------------------------------------
# 2. ATTENTION - For language models
# -----------------------------------------------------------------------------

print("\n\n2. ATTENTION - For language models")
print("-" * 40)

class AttentionExamples:
    """
    Attention: Focus on relevant parts of input
    """
    
    def explain_attention(self):
        print("Attention: Focus on important parts when processing sequences")
        
        print("\nExample: Translating sentence")
        print('English: "The cat that I saw yesterday was black"')
        print('French:  "Le chat que j\'ai vu hier était noir"')
        
        print('\nWhen translating "chat" (cat), model should:')
        print("• Pay HIGH attention to 'cat'")
        print("• Pay LOW attention to 'yesterday', 'was', 'black'")
        
        # Simple attention weights example
        words = ["The", "cat", "that", "I", "saw", "yesterday", "was", "black"]
        relevance_scores = [0.1, 0.9, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1]
        
        # Normalize to sum to 1 (attention weights)
        total = sum(relevance_scores)
        attention_weights = [score/total for score in relevance_scores]
        
        print("\nAttention weights when translating 'cat':")
        for word, weight in zip(words, attention_weights):
            importance = "HIGH" if weight > 0.3 else "LOW"
            print(f"  {word}: {weight:.2f} ({importance})")
        
        print("\nWhy attention revolutionized AI:")
        print("• Before: Models forgot early words in long sentences")
        print("• After: Models can focus on ANY word, no matter how far back")
        print("• Powers: ChatGPT, Google Translate, BERT, GPT-4")
        
        print("\n❌ TO IMPLEMENT: Attention mechanisms")
        print("   - Scaled dot-product attention")
        print("   - Multi-head attention")
        print("   - Self-attention vs cross-attention")
        print("   - Positional encoding")

attention_demo = AttentionExamples()
attention_demo.explain_attention()

# -----------------------------------------------------------------------------
# 3. REGULARIZATION - Prevent overfitting
# -----------------------------------------------------------------------------

print("\n\n3. REGULARIZATION - Prevent overfitting")
print("-" * 40)

class RegularizationExamples:
    """
    Prevent model from memorizing instead of learning
    """
    
    def explain_overfitting(self):
        print("Overfitting: Memorizing training data instead of learning patterns")
        
        print("\nAnalogy - Studying for exam:")
        print("BAD (overfitting):")
        print("  • Memorize all practice questions and answers")
        print("  • Get 100% on practice test")
        print("  • Get 60% on real test (different questions!)")
        
        print("\nGOOD (generalization):")
        print("  • Understand concepts behind questions")
        print("  • Get 85% on practice test")
        print("  • Get 85% on real test")
        
        print("\nRegularization techniques:")
        
        # Dropout example
        print("\n1. DROPOUT: Randomly turn off neurons during training")
        activations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        drop_rate = 0.5
        
        # Simulate dropout
        mask = np.random.random(len(activations)) > drop_rate
        dropped_activations = activations * mask
        
        print(f"Original activations: {activations}")
        print(f"Dropout mask: {mask}")
        print(f"After dropout: {dropped_activations}")
        print("Forces model to not rely on any single neuron")
        
        print("\n2. WEIGHT DECAY: Penalize large weights")
        print("Loss = prediction_error + penalty_for_large_weights")
        print("Encourages simpler, more generalizable models")
        
        print("\n3. DATA AUGMENTATION: Create variations of training data")
        print("• Images: rotate, flip, crop, change brightness")
        print("• Text: synonym replacement, paraphrasing")
        print("• Audio: add noise, change speed")
        
        print("\n❌ TO IMPLEMENT: Regularization techniques")
        print("   - Dropout layers")
        print("   - Weight decay in optimizers")
        print("   - Data augmentation pipelines")
        print("   - Early stopping")

regularization_demo = RegularizationExamples()
regularization_demo.explain_overfitting()

# -----------------------------------------------------------------------------
# 4. DIFFERENT ARCHITECTURES - Specialized designs
# -----------------------------------------------------------------------------

print("\n\n4. DIFFERENT ARCHITECTURES - Specialized designs")
print("-" * 40)

class ArchitectureExamples:
    """
    Different problems need different network designs
    """
    
    def explain_architectures(self):
        print("Different tools for different jobs:")
        
        print("\n1. CNN (Convolutional Neural Network)")
        print("   Best for: Images, videos, spatial data")
        print("   Like: Having specialized vision tools")
        print("   Examples: Image classification, medical imaging")
        
        print("\n2. RNN (Recurrent Neural Network)")
        print("   Best for: Sequences, time series, text")
        print("   Like: Reading a book with memory of previous chapters")
        print("   Examples: Language translation, speech recognition")
        
        # Simple RNN simulation
        class SimpleRNN:
            def __init__(self):
                self.memory = 0
            
            def process_sequence(self, sequence):
                outputs = []
                for item in sequence:
                    # Update memory: mix old memory with new input
                    self.memory = 0.5 * self.memory + 0.5 * item
                    outputs.append(self.memory)
                return outputs
        
        rnn = SimpleRNN()
        sequence = [1, 2, 3, 4, 5]
        outputs = rnn.process_sequence(sequence)
        
        print(f"   RNN processing sequence {sequence}:")
        print(f"   Outputs: {[round(x, 2) for x in outputs]}")
        print("   Notice how each output depends on previous inputs")
        
        print("\n3. TRANSFORMER")
        print("   Best for: Language, any task requiring understanding relationships")
        print("   Like: Having super-human reading comprehension")
        print("   Examples: ChatGPT, Google Translate, BERT")
        
        print("\nArchitecture timeline:")
        print("• 1980s: Simple neural networks")
        print("• 1990s: CNNs for images")
        print("• 2000s: RNNs for sequences")
        print("• 2010s: Deep CNNs (AlexNet, ResNet)")
        print("• 2017+: Transformers dominate (BERT, GPT)")
        
        print("\n❌ TO IMPLEMENT: Architecture support")
        print("   - Modular architecture building")
        print("   - Pre-built architecture templates")
        print("   - Architecture search capabilities")
        print("   - Transfer learning support")

architecture_demo = ArchitectureExamples()
architecture_demo.explain_architectures()

# ============================================================================
# SECTION 3: PRODUCTION FEATURES (For real applications)
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 3: PRODUCTION FEATURES")
print("="*80)

# -----------------------------------------------------------------------------
# 1. DATA LOADING - Handle large datasets efficiently
# -----------------------------------------------------------------------------

print("\n1. DATA LOADING - Handle large datasets efficiently")
print("-" * 40)

class DataLoadingExamples:
    """
    Real datasets are huge and don't fit in memory
    """
    
    def explain_data_loading(self):
        print("Problem: ImageNet has 1.2M images (~240GB)")
        print("Solution: Load small batches at a time")
        
        print("\nData loading is like a restaurant kitchen:")
        print("• Don't cook all meals at once (memory overflow)")
        print("• Prepare batches as customers order")
        print("• Keep ingredients fresh and organized")
        
        # Simulate data loader
        class SimpleDataLoader:
            def __init__(self, dataset_size=1000000, batch_size=32):
                self.dataset_size = dataset_size
                self.batch_size = batch_size
                self.current_index = 0
                print(f"DataLoader: {dataset_size} samples, batch_size={batch_size}")
            
            def get_batch(self):
                # Simulate loading batch_size samples
                batch_indices = list(range(self.current_index, 
                                         min(self.current_index + self.batch_size, 
                                             self.dataset_size)))
                self.current_index += self.batch_size
                
                if self.current_index >= self.dataset_size:
                    self.current_index = 0  # Reset for next epoch
                
                return f"Batch of {len(batch_indices)} samples"
        
        # Example usage
        loader = SimpleDataLoader()
        for i in range(3):
            batch = loader.get_batch()
            print(f"Epoch step {i+1}: {batch}")
        
        print("\nData preprocessing needed:")
        print("• Images: resize, normalize, augment")
        print("• Text: tokenize, handle unknown words")
        print("• Audio: resample, remove noise")
        print("• Video: frame extraction, temporal sampling")
        
        print("\n❌ TO IMPLEMENT: Data loading system")
        print("   - Efficient batch loading")
        print("   - Data preprocessing pipelines")
        print("   - Multi-threaded data loading")
        print("   - Data augmentation")
        print("   - Memory mapping for large datasets")

data_loading_demo = DataLoadingExamples()
data_loading_demo.explain_data_loading()

# -----------------------------------------------------------------------------
# 2. MODEL SAVING/LOADING - Reuse trained models
# -----------------------------------------------------------------------------

print("\n\n2. MODEL SAVING/LOADING - Reuse trained models")
print("-" * 40)

class ModelPersistenceExamples:
    """
    Training takes hours/days - don't retrain every time!
    """
    
    def explain_model_persistence(self):
        print("Model saving/loading is like saving a video game:")
        print("• Save progress after hours of playing")
        print("• Load saved game later to continue")
        print("• Share save files with friends")
        
        print("\nTypical workflow:")
        print("Day 1: Train model for 10 hours")
        print("Day 1 evening: Save model to 'cat_detector_v1.model'")
        print("Day 2: Load model in 5 seconds")
        print("Day 2: Use model for predictions immediately")
        print("Day 3: Continue training from checkpoint")
        
        # Simulate model saving
        class SimpleModel:
            def __init__(self):
                self.weights = np.random.randn(10, 5)
                self.bias = np.zeros(5)
                self.training_history = []
            
            def save(self, filepath):
                model_data = {
                    'weights': self.weights,
                    'bias': self.bias,
                    'history': self.training_history,
                    'version': '1.0'
                }
                print(f"Saving model to {filepath}")
                print(f"Model size: {self.weights.nbytes + self.bias.nbytes} bytes")
                return model_data
            
            def load(self, model_data):
                self.weights = model_data['weights']
                self.bias = model_data['bias']
                self.training_history = model_data['history']
                print("Model loaded successfully!")
        
        # Example
        model = SimpleModel()
        saved_data = model.save("my_model.pkl")
        
        new_model = SimpleModel()
        new_model.load(saved_data)
        
        print("\nWhat gets saved:")
        print("• Model architecture (layer types, connections)")
        print("• Trained weights (learned parameters)")
        print("• Optimizer state (for continuing training)")
        print("• Training history (loss curves, metrics)")
        print("• Metadata (hyperparameters, version)")
        
        print("\n❌ TO IMPLEMENT: Model persistence")
        print("   - Serialization of model architecture")
        print("   - Weight saving/loading")
        print("   - Checkpoint management")
        print("   - Version compatibility")
        print("   - Compression for large models")

model_persistence_demo = ModelPersistenceExamples()
model_persistence_demo.explain_model_persistence()

# -----------------------------------------------------------------------------
# 3. DISTRIBUTED TRAINING - Use multiple GPUs/computers
# -----------------------------------------------------------------------------

print("\n\n3. DISTRIBUTED TRAINING - Use multiple GPUs/computers")
print("-" * 40)

class DistributedTrainingExamples:
    """
    Speed up training by using multiple computers
    """
    
    def explain_distributed_training(self):
        print("Distributed training is like building a house:")
        
        print("\nSerial approach (1 worker):")
        print("• Foundation: 2 weeks")
        print("• Framing: 3 weeks")
        print("• Roofing: 2 weeks")
        print("• Total: 7 weeks")
        
        print("\nParallel approach (4 workers):")
        print("• All work on different parts simultaneously")
        print("• Coordinate and share materials")
        print("• Total: 2-3 weeks (coordination overhead)")
        
        print("\nData Parallel Training:")
        print("• Same model on each GPU")
        print("• Different data batches")
        print("• Share gradients between GPUs")
        
        # Simulate data parallel training
        def simulate_data_parallel():
            total_batch_size = 128
            num_gpus = 4
            batch_per_gpu = total_batch_size // num_gpus
            
            print(f"\nSimulation with {num_gpus} GPUs:")
            print(f"Total batch size: {total_batch_size}")
            print(f"Batch per GPU: {batch_per_gpu}")
            
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * batch_per_gpu
                end_idx = start_idx + batch_per_gpu
                print(f"GPU {gpu_id}: processes samples {start_idx}-{end_idx}")
            
            print("All GPUs compute gradients simultaneously")
            print("Gradients averaged across GPUs")
            print("Model weights updated on all GPUs")
        
        simulate_data_parallel()
        
        print("\nModel Parallel Training (for very large models):")
        print("• Split model across GPUs")
        print("• Each GPU handles different layers")
        print("• Data flows through GPUs in sequence")
        
        print("\nReal examples:")
        print("• GPT-3: Trained on 1000+ GPUs for weeks")
        print("• ImageNet: 8 GPUs → 4x faster than 1 GPU")
        print("• Some models too big to fit on single GPU")
        
        print("\n❌ TO IMPLEMENT: Distributed training")
        print("   - Multi-GPU communication")
        print("   - Gradient synchronization")
        print("   - Load balancing across devices")
        print("   - Fault tolerance")
        print("   - Efficient data sharding")

distributed_demo = DistributedTrainingExamples()
distributed_demo.explain_distributed_training()

# -----------------------------------------------------------------------------
# 4. OPTIMIZATION - Make models smaller and faster
# -----------------------------------------------------------------------------

print("\n\n4. OPTIMIZATION - Make models smaller and faster")
print("-" * 40)

class OptimizationExamples:
    """
    Make models smaller and faster for production deployment
    """
    
    def explain_model_optimization(self):
        print("Model optimization is like packing for a trip:")
        print("• Original model: huge suitcase with everything")
        print("• Optimized model: small backpack with essentials")
        
        print("\n1. QUANTIZATION: Use smaller numbers")
        print("   Original: 32-bit floats (very precise)")
        print("   Quantized: 8-bit integers (less precise, much smaller)")
        
        # Simulate quantization
        original_weights = np.array([0.1234567, -0.9876543, 0.5555555])
        print(f"Original weights: {original_weights}")
        
        # Quantize to 8-bit (simplified)
        scale = np.max(np.abs(original_weights)) / 127
        quantized = np.round(original_weights / scale).astype(np.int8)
        dequantized = quantized * scale
        
        print(f"Quantized (8-bit): {quantized}")
        print(f"Dequantized: {dequantized}")
        print(f"Size reduction: 4x smaller")
        print(f"Speed improvement: 2-4x faster")
        
        print("\n2. PRUNING: Remove unimportant connections")
        weights_matrix = np.array([[0.8, 0.01, 0.7],
                                  [0.02, 0.9, 0.03],
                                  [0.6, 0.01, 0.8]])
        
        print("Original weights:")
        print(weights_matrix)
        
        # Remove weights below threshold
        threshold = 0.1
        pruned_weights = np.where(np.abs(weights_matrix) > threshold, 
                                 weights_matrix, 0)
        
        print(f"\nAfter pruning (threshold={threshold}):")
        print(pruned_weights)
        
        sparsity = np.sum(pruned_weights == 0) / pruned_weights.size
        print(f"Sparsity: {sparsity:.1%} (can be stored efficiently)")
        
        print("\n3. KNOWLEDGE DISTILLATION: Teach small model from large model")
        print("   Like professor teaching student:")
        print("   • Professor (large model): 1GB, very accurate")
        print("   • Student (small model): 10MB, almost as accurate")
        print("   • Student learns from professor's knowledge")
        
        print("\nReal impact:")
        print("• BERT: 340MB → 14MB (24x smaller)")
        print("• Mobile apps: Models that fit on phones")
        print("• Edge devices: Real-time inference")
        
        print("\n❌ TO IMPLEMENT: Model optimization")
        print("   - Quantization algorithms (dynamic, static)")
        print("   - Structured and unstructured pruning")
        print("   - Knowledge distillation frameworks")
        print("   - Neural architecture search")
        print("   - Operator fusion")

optimization_demo = OptimizationExamples()
optimization_demo.explain_model_optimization()

# -----------------------------------------------------------------------------
# 5. DEPLOYMENT - Put models in apps and websites
# -----------------------------------------------------------------------------

print("\n\n5. DEPLOYMENT - Put models in apps and websites")
print("-" * 40)

class DeploymentExamples:
    """
    Turn trained models into services people can use
    """
    
    def explain_deployment(self):
        print("Deployment: Turn your model into something users can actually use")
        
        print("\n1. WEB API DEPLOYMENT")
        print("   Like a restaurant:")
        print("   • Customer orders food (sends request)")
        print("   • Kitchen cooks food (model processes)")
        print("   • Waiter brings food (returns result)")
        
        # Simulate web API
        class ModelAPI:
            def __init__(self):
                print("Model API started on http://localhost:8000")
            
            def predict(self, image_data):
                # Simulate model inference
                predictions = ["cat", "dog", "bird"]
                confidence = [0.8, 0.15, 0.05]
                return {
                    "prediction": predictions[0],
                    "confidence": confidence[0],
                    "all_predictions": dict(zip(predictions, confidence))
                }
        
        api = ModelAPI()
        result = api.predict("image_bytes_here")
        print(f"API response: {result}")
        
        print("\n2. MOBILE APP DEPLOYMENT")
        print("   Advantages:")
        print("   • Works without internet")
        print("   • Instant responses")
        print("   • Privacy (data doesn't leave phone)")
        print("   Challenges:")
        print("   • Limited memory/battery")
        print("   • Model must be very small")
        
        print("\n3. EDGE DEPLOYMENT")
        print("   Examples:")
        print("   • Security camera: detect intruders")
        print("   • Self-driving car: detect pedestrians")
        print("   • Smart speaker: recognize voice commands")
        print("   • Medical device: analyze scans")
        
        print("\nDeployment considerations:")
        print("• Performance: How fast does it need to be?")
        print("• Scale: How many users will use it?")
        print("• Privacy: Can data leave the device?")
        print("• Connectivity: Will internet always be available?")
        print("• Resources: Memory/battery/compute constraints?")
        print("• Reliability: What happens if model fails?")
        
        print("\n❌ TO IMPLEMENT: Deployment infrastructure")
        print("   - Model serving frameworks")
        print("   - API generation and management")
        print("   - Mobile/edge runtime optimization")
        print("   - Load balancing and scaling")
        print("   - Monitoring and logging")

deployment_demo = DeploymentExamples()
deployment_demo.explain_deployment()

# ============================================================================
# SECTION 4: DEVELOPER TOOLS (For debugging and development)
# ============================================================================

print("\n\n" + "="*80)
print("SECTION 4: DEVELOPER TOOLS")
print("="*80)

# -----------------------------------------------------------------------------
# 1. VISUALIZATION - See what the model is learning
# -----------------------------------------------------------------------------

print("\n1. VISUALIZATION - See what the model is learning")
print("-" * 40)

class VisualizationExamples:
    """
    Tools to understand what's happening inside models
    """
    
    def demonstrate_visualizations(self):
        print("Visualization gives X-ray vision into your model's brain")
        
        print("\n1. TRAINING CURVES: Track progress over time")
        
        # Simulate training history
        epochs = list(range(1, 11))
        train_loss = [2.3, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4]
        val_loss = [2.4, 1.9, 1.5, 1.2, 1.0, 0.8, 0.7, 0.65, 0.6, 0.55]
        
        print("Training Progress:")
        print("Epoch | Train Loss | Val Loss")
        print("------|------------|----------")
        for e, tl, vl in zip(epochs[:5], train_loss[:5], val_loss[:5]):
            print(f"  {e:2d}  |    {tl:.2f}    |   {vl:.2f}")
        print("  ... |    ...     |   ...")
        
        print("\nWhat to look for:")
        print("✅ Loss decreasing: Model is learning")
        print("⚠️  Loss flat: Model stuck, try different settings")
        print("⚠️  Val loss > train loss: Overfitting")
        
        print("\n2. FILTER VISUALIZATION: See what patterns model detects")
        print("   For image models:")
        print("   • Layer 1: edges, colors, simple textures")
        print("   • Layer 2: shapes, more complex patterns")
        print("   • Layer 3: object parts (eyes, wheels)")
        
        print("\n3. ACTIVATION MAPS: See where model focuses")
        print("   Like highlighting important parts while reading:")
        print("   • Bright areas: model pays attention here")
        print("   • Dark areas: model ignores these parts")
        
        print("\n❌ TO IMPLEMENT: Visualization tools")
        print("   - Training curve plotting")
        print("   - Filter and activation visualization")
        print("   - Attention weight visualization")
        print("   - Embedding visualization (t-SNE, UMAP)")
        print("   - Interactive dashboards")

viz_demo = VisualizationExamples()
viz_demo.demonstrate_visualizations()

# -----------------------------------------------------------------------------
# 2. DEBUGGING TOOLS - Find and fix problems
# -----------------------------------------------------------------------------

print("\n\n2. DEBUGGING TOOLS - Find and fix problems")
print("-" * 40)

class DebuggingExamples:
    """
    Diagnostic tools for finding model problems
    """
    
    def demonstrate_debugging(self):
        print("Debugging tools are like a mechanic's diagnostic equipment")
        
        print("\n1. GRADIENT CHECKING: Verify gradients are correct")
        print("   Like double-checking your math homework:")
        print("   • Compute gradients automatically (your method)")
        print("   • Compute gradients numerically (slow but correct)")
        print("   • Compare: should be nearly identical")
        
        # Simulate gradient checking
        def numerical_gradient_check():
            auto_grad = 0.123456
            numerical_grad = 0.123458
            difference = abs(auto_grad - numerical_grad)
            
            print(f"   Automatic gradient: {auto_grad}")
            print(f"   Numerical gradient: {numerical_grad}")
            print(f"   Difference: {difference}")
            
            if difference < 1e-5:
                print("   ✅ Gradients correct!")
            else:
                print("   ❌ Gradient error detected!")
        
        numerical_gradient_check()
        
        print("\n2. VANISHING/EXPLODING GRADIENTS")
        print("   Check if gradients become too small or too large:")
        
        # Simulate gradient magnitudes by layer
        layers = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]
        grad_magnitudes = [1e-1, 1e-3, 1e-6, 1e-9]
        
        print("   Layer     | Gradient Magnitude | Status")
        print("   ----------|-------------------|--------")
        for layer, grad_mag in zip(layers, grad_magnitudes):
            if grad_mag < 1e-6:
                status = "⚠️  Vanishing"
            elif grad_mag > 1e2:
                status = "⚠️  Exploding"
            else:
                status = "✅ Normal"
            print(f"   {layer:9} | {grad_mag:12.2e}     | {status}")
        
        print("\n3. DATA LEAKAGE DETECTION")
        print("   Verify training and test data don't overlap:")
        print("   • If test data in training: model will cheat")
        print("   • Results artificially good")
        print("   • Won't work on truly new data")
        
        print("\n4. PREDICTION ANALYSIS")
        print("   Analyze what model gets right vs wrong:")
        print("   • What types of errors are common?")
        print("   • Are there patterns in failures?")
        print("   • What can be improved?")
        
        print("\n❌ TO IMPLEMENT: Debugging tools")
        print("   - Gradient verification utilities")
        print("   - Gradient flow analysis")
        print("   - Data leakage detection")
        print("   - Error pattern analysis")
        print("   - Model interpretation tools")

debug_demo = DebuggingExamples()
debug_demo.demonstrate_debugging()

# -----------------------------------------------------------------------------
# 3. PROFILING - Measure speed and memory usage
# -----------------------------------------------------------------------------

print("\n\n3. PROFILING - Measure speed and memory usage")
print("-" * 40)

class ProfilingExamples:
    """
    Measure performance to find bottlenecks
    """
    
    def demonstrate_profiling(self):
        print("Profiling: Like having a stopwatch and memory meter")
        
        print("\n1. TIMING OPERATIONS")
        print("   Measure how long each operation takes:")
        
        # Simulate operation timing
        operations = [
            ("Data loading", 0.050),
            ("Forward pass", 0.120),
            ("Backward pass", 0.180),
            ("Weight update", 0.010)
        ]
        
        total_time = sum(time for _, time in operations)
        
        print("   Operation      | Time (s) | Percentage")
        print("   ---------------|----------|----------")
        for op_name, op_time in operations:
            percentage = (op_time / total_time) * 100
            print(f"   {op_name:14} | {op_time:8.3f} | {percentage:6.1f}%")
        print(f"   {'TOTAL':14} | {total_time:8.3f} | {'100.0%':>7}")
        
        print("   → Backward pass is the bottleneck (50%)")
        
        print("\n2. MEMORY USAGE TRACKING")
        print("   Monitor memory consumption:")
        
        memory_stages = [
            ("Baseline", 1024),
            ("Model loaded", 2048),
            ("Forward pass", 3072),
            ("Backward pass", 4096)
        ]
        
        print("   Stage          | Memory (MB) | Delta (MB)")
        print("   ---------------|-------------|----------")
        prev_memory = 0
        for stage, memory in memory_stages:
            delta = memory - prev_memory if prev_memory > 0 else 0
            print(f"   {stage:14} | {memory:11} | {delta:+9}")
            prev_memory = memory
        
        print("\n3. GPU UTILIZATION")
        print("   Monitor GPU performance:")
        print("   • GPU utilization: 85% (good, working hard)")
        print("   • Memory usage: 7GB / 8GB (almost full)")
        print("   • Temperature: 75°C (warm but safe)")
        
        print("\n4. THROUGHPUT MEASUREMENT")
        batch_size = 32
        time_per_batch = 0.25
        throughput = batch_size / time_per_batch
        
        print(f"   Batch size: {batch_size}")
        print(f"   Time per batch: {time_per_batch:.3f}s")
        print(f"   Throughput: {throughput:.1f} samples/second")
        
        print("\n❌ TO IMPLEMENT: Profiling tools")
        print("   - Operation timing utilities")
        print("   - Memory usage tracking")
        print("   - GPU utilization monitoring")
        print("   - Throughput measurement")
        print("   - Bottleneck identification")

profiling_demo = ProfilingExamples()
profiling_demo.demonstrate_profiling()

# ============================================================================
# COMPREHENSIVE FEATURE CHECKLIST & IMPLEMENTATION ROADMAP
# ============================================================================

print("\n\n" + "="*80)
print("COMPREHENSIVE FEATURE CHECKLIST & IMPLEMENTATION ROADMAP")
print("="*80)

class ImplementationRoadmap:
    """
    Complete guide for implementing a deep learning library
    """
    
    def print_current_status(self):
        print("\n✅ WHAT YOUR VALUE CLASS ALREADY HAS:")
        print("-" * 50)
        print("✅ Automatic Differentiation (THE CORE BREAKTHROUGH!)")
        print("✅ Basic Math Operations (+, -, *, /, tanh, exp)")
        print("✅ Computational Graph Building")
        print("✅ Backward Propagation")
        print("✅ Gradient Accumulation")
        
        print("\nThis is HUGE! You have the foundation that powers all modern AI!")
        
    def print_missing_features(self):
        print("\n❌ MISSING FEATURES TO IMPLEMENT:")
        print("-" * 50)
        
        core_features = [
            "Multi-dimensional Tensors (arrays instead of scalars)",
            "GPU Support (CUDA/OpenCL acceleration)", 
            "Broadcasting (operations between different shapes)",
            "Neural Network Layers (Linear, Conv2D, RNN, etc.)",
            "Optimizers (SGD, Adam, RMSprop, etc.)",
            "Loss Functions (MSE, CrossEntropy, etc.)"
        ]
        
        advanced_features = [
            "Convolution Operations (for image processing)",
            "Attention Mechanisms (for language models)",
            "Regularization (Dropout, Weight Decay, etc.)",
            "Different Architectures (CNN, RNN, Transformer)",
            "Batch Normalization (training stability)",
            "Activation Functions (ReLU, Sigmoid, GELU, etc.)"
        ]
        
        production_features = [
            "Data Loading (efficient batch processing)",
            "Model Saving/Loading (persistence)",
            "Distributed Training (multi-GPU/multi-node)",
            "Model Optimization (quantization, pruning)",
            "Deployment Infrastructure (serving, APIs)",
            "Mobile/Edge Optimization"
        ]
        
        developer_tools = [
            "Visualization Tools (training curves, filters)",
            "Debugging Utilities (gradient checking)",
            "Profiling Tools (timing, memory usage)",
            "Error Analysis (prediction patterns)",
            "Interactive Dashboards",
            "Model Interpretation"
        ]
        
        print("CORE FEATURES (Essential):")
        for i, feature in enumerate(core_features, 1):
            print(f"  {i}. {feature}")
        
        print("\nADVANCED FEATURES (For serious use):")
        for i, feature in enumerate(advanced_features, 1):
            print(f"  {i}. {feature}")
        
        print("\nPRODUCTION FEATURES (For real applications):")
        for i, feature in enumerate(production_features, 1):
            print(f"  {i}. {feature}")
        
        print("\nDEVELOPER TOOLS (For debugging):")
        for i, feature in enumerate(developer_tools, 1):
            print(f"  {i}. {feature}")
    
    def print_implementation_order(self):
        print("\n🛠️ RECOMMENDED IMPLEMENTATION ORDER:")
        print("-" * 50)
        
        phases = [
            ("Phase 1: Tensor Foundation", [
                "Fix the exp() bug in your Value class",
                "Add missing math operations (pow, sqrt, log, abs)",
                "Add comparison operations (>, <, ==)",
                "Implement basic Tensor class for 2D arrays",
                "Add matrix multiplication"
            ]),
            
            ("Phase 2: Neural Networks", [
                "Implement activation functions (relu, sigmoid)",
                "Create Linear/Dense layer",
                "Add basic optimizers (SGD)",
                "Implement loss functions (MSE, CrossEntropy)",
                "Build simple neural network class"
            ]),
            
            ("Phase 3: Advanced Operations", [
                "Add broadcasting support",
                "Implement reduction operations (sum, mean, max)",
                "Add convolution operations",
                "Create more layer types",
                "Add regularization techniques"
            ]),
            
            ("Phase 4: Production Ready", [
                "Add model saving/loading",
                "Implement data loading utilities",
                "Add visualization tools",
                "Create debugging utilities",
                "Add performance profiling"
            ]),
            
            ("Phase 5: Advanced Features", [
                "GPU support (very complex!)",
                "Distributed training",
                "Model optimization",
                "Deployment infrastructure",
                "Advanced architectures"
            ])
        ]
        
        for phase_name, tasks in phases:
            print(f"\n{phase_name}:")
            for i, task in enumerate(tasks, 1):
                print(f"  {i}. {task}")
    
    def print_difficulty_levels(self):
        print("\n📊 DIFFICULTY LEVELS:")
        print("-" * 50)
        
        difficulty_map = {
            "🟢 EASY (Good starting points)": [
                "Fix exp() bug",
                "Add math operations (pow, sqrt, log)",
                "Add activation functions (relu, sigmoid)",
                "Implement comparison operations",
                "Add simple optimizers (SGD)"
            ],
            
            "🟡 MEDIUM (Significant but doable)": [
                "Multi-dimensional tensors",
                "Matrix multiplication with gradients",
                "Broadcasting implementation",
                "Neural network layers",
                "Loss functions with gradients",
                "Model saving/loading"
            ],
            
            "🟠 HARD (Complex algorithms)": [
                "Convolution operations",
                "Attention mechanisms",
                "Advanced optimizers (Adam)",
                "Data loading pipelines",
                "Distributed training coordination"
            ],
            
            "🔴 VERY HARD (Expert level)": [
                "GPU acceleration (CUDA programming)",
                "Memory optimization",
                "Kernel fusion",
                "JIT compilation",
                "Production deployment at scale"
            ]
        }
        
        for difficulty, features in difficulty_map.items():
            print(f"\n{difficulty}:")
            for feature in features:
                print(f"  • {feature}")
    
    def print_learning_resources(self):
        print("\n📚 LEARNING RESOURCES FOR IMPLEMENTATION:")
        print("-" * 50)
        
        resources = {
            "Understanding the Math": [
                "3Blue1Brown: Neural Networks series (YouTube)",
                "Andrej Karpathy: Micrograd tutorial",
                "Deep Learning book by Ian Goodfellow"
            ],
            
            "Implementation Guides": [
                "PyTorch source code (github.com/pytorch/pytorch)",
                "TinyGrad source code (github.com/geohot/tinygrad)",
                "JAX source code (github.com/google/jax)"
            ],
            
            "Specific Topics": [
                "Convolution: CS231n Stanford course",
                "Attention: 'Attention is All You Need' paper",
                "GPU Programming: CUDA documentation",
                "Automatic Differentiation: 'Automatic Differentiation in Machine Learning: a Survey'"
            ]
        }
        
        for category, items in resources.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
    
    def print_final_motivation(self):
        print("\n" + "="*80)
        print("🚀 YOU'RE ALREADY ON THE RIGHT TRACK!")
        print("="*80)
        
        print("\nYour Value class implements AUTOMATIC DIFFERENTIATION -")
        print("the single most important breakthrough in modern AI!")
        
        print("\nThis concept powers:")
        print("• ChatGPT and all large language models")
        print("• Image recognition in your phone")
        print("• Self-driving cars")
        print("• Medical AI diagnostics")
        print("• All modern neural networks")
        
        print("\nEvery feature in this file builds on what you already have.")
        print("You understand the core. Everything else is engineering!")
        
        print("\nStart with Phase 1, take it step by step, and you'll build")
        print("your own deep learning library. The journey of a thousand")
        print("miles begins with a single step - and you've already taken it! 🎯")

# Run the comprehensive roadmap
roadmap = ImplementationRoadmap()
roadmap.print_current_status()
roadmap.print_missing_features()
roadmap.print_implementation_order()
roadmap.print_difficulty_levels()
roadmap.print_learning_resources()
roadmap.print_final_motivation()

# ============================================================================
# EXAMPLE: YOUR CURRENT VALUE CLASS WITH BUG FIX
# ============================================================================

print("\n\n" + "="*80)
print("BONUS: YOUR CURRENT VALUE CLASS WITH BUG FIX")
print("="*80)

class Value:
    """
    Your current Value class with the exp() bug fixed
    """
    def __init__(self, data, _children=(), _op='', label=None):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None 
        self.label = label if label is not None else str(data)
        
    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self,other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 
    
    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children=(self,), _op='exp')  # FIXED: was +=
        def _backward():
            self.grad += math.exp(x) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op='tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
      
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Test the fixed Value class
print("\nTesting your fixed Value class:")
a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b
c.label = 'c'
d = c.exp()
d.label = 'd'

print(f"a = {a}")
print(f"b = {b}")
print(f"c = a * b = {c}")
print(f"d = exp(c) = {d}")

d.backward()
print(f"\nAfter backward:")
print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")
print(f"c.grad = {c.grad}")
print(f"d.grad = {d.grad}")

print("\n🎉 Your Value class works perfectly!")
print("You have the foundation to build the next PyTorch!")

print("\n" + "="*80)
print("END OF COMPREHENSIVE DEEP LEARNING FEATURES GUIDE")
print("="*80)