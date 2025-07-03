"""
Tensor Broadcasting and Memory Layout Deep Dive

This covers:
1. How broadcasting actually works mathematically
2. Memory layout optimization (row-major vs column-major)
3. Stride calculations and memory access patterns
4. Broadcasting in gradients (the tricky part!)
5. Cache-friendly operations
6. Real-world performance implications
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import time
from collections import namedtuple

# ============================================================================
# PART 1: UNDERSTANDING MEMORY LAYOUT AND STRIDES
# ============================================================================

class TensorLayout:
    """
    Detailed tensor layout information showing how data is actually stored in memory
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32):
        self.shape = shape
        self.dtype = dtype
        self.itemsize = dtype().itemsize  # bytes per element
        self.ndim = len(shape)
        
        # Calculate strides for row-major (C-style) layout
        self.strides = self._calculate_strides(shape, self.itemsize)
        self.size = np.prod(shape)
        self.nbytes = self.size * self.itemsize
    
    def _calculate_strides(self, shape: Tuple[int, ...], itemsize: int) -> Tuple[int, ...]:
        """
        Calculate strides for row-major layout
        Stride[i] = number of bytes to skip to get to next element along dimension i
        """
        if not shape:
            return ()
        
        strides = [itemsize]
        for i in range(len(shape) - 2, -1, -1):
            strides.insert(0, strides[0] * shape[i + 1])
        
        return tuple(strides)
    
    def get_memory_offset(self, indices: Tuple[int, ...]) -> int:
        """
        Calculate memory offset for given indices
        offset = sum(index[i] * stride[i] for i in dimensions)
        """
        assert len(indices) == self.ndim, f"Expected {self.ndim} indices, got {len(indices)}"
        
        offset = 0
        for i, (index, stride) in enumerate(zip(indices, self.strides)):
            assert 0 <= index < self.shape[i], f"Index {index} out of bounds for dimension {i} (size {self.shape[i]})"
            offset += index * stride
        
        return offset
    
    def demonstrate_memory_layout(self):
        """Show how elements are laid out in memory"""
        print(f"Tensor Layout Analysis:")
        print(f"  Shape: {self.shape}")
        print(f"  Strides (bytes): {self.strides}")
        print(f"  Element size: {self.itemsize} bytes")
        print(f"  Total memory: {self.nbytes} bytes")
        
        if self.ndim <= 3:  # Only show for small tensors
            print(f"\n  Memory layout (showing byte offsets):")
            
            if self.ndim == 2:
                for i in range(min(4, self.shape[0])):
                    row = []
                    for j in range(min(6, self.shape[1])):
                        offset = self.get_memory_offset((i, j))
                        row.append(f"{offset:3d}")
                    if self.shape[1] > 6:
                        row.append("...")
                    print(f"    Row {i}: [{', '.join(row)}]")
                if self.shape[0] > 4:
                    print("    ...")
            
            elif self.ndim == 3:
                for i in range(min(2, self.shape[0])):
                    print(f"    Matrix {i}:")
                    for j in range(min(3, self.shape[1])):
                        row = []
                        for k in range(min(4, self.shape[2])):
                            offset = self.get_memory_offset((i, j, k))
                            row.append(f"{offset:3d}")
                        print(f"      [{', '.join(row)}]")

def compare_memory_layouts():
    """Compare row-major vs column-major memory access patterns"""
    
    print("=" * 60)
    print("MEMORY LAYOUT COMPARISON")
    print("=" * 60)
    
    # Create test matrices
    size = 1000
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    print(f"Testing {size}x{size} matrices")
    print(f"Matrix A strides: {A.strides} (row-major)")
    
    # Create column-major version
    A_col = np.asfortranarray(A)  # Column-major
    print(f"Matrix A_col strides: {A_col.strides} (column-major)")
    
    # Test row-wise access (cache-friendly for row-major)
    print(f"\n1. Row-wise access:")
    start_time = time.time()
    for i in range(0, size, 10):  # Sample every 10th row
        row_sum = np.sum(A[i, :])  # Access entire row
    row_major_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(0, size, 10):
        row_sum = np.sum(A_col[i, :])  # Access entire row in column-major
    col_major_time = time.time() - start_time
    
    print(f"  Row-major: {row_major_time*1000:.2f}ms")
    print(f"  Column-major: {col_major_time*1000:.2f}ms")
    print(f"  Speedup: {col_major_time/row_major_time:.1f}x")
    
    # Test column-wise access (cache-friendly for column-major)
    print(f"\n2. Column-wise access:")
    start_time = time.time()
    for j in range(0, size, 10):  # Sample every 10th column
        col_sum = np.sum(A[:, j])  # Access entire column
    row_major_time = time.time() - start_time
    
    start_time = time.time()
    for j in range(0, size, 10):
        col_sum = np.sum(A_col[:, j])  # Access entire column in column-major
    col_major_time = time.time() - start_time
    
    print(f"  Row-major: {row_major_time*1000:.2f}ms")
    print(f"  Column-major: {col_major_time*1000:.2f}ms")
    print(f"  Speedup: {row_major_time/col_major_time:.1f}x")

# ============================================================================
# PART 2: BROADCASTING MECHANICS
# ============================================================================

class BroadcastingEngine:
    """
    Detailed implementation of broadcasting rules with step-by-step explanation
    """
    
    @staticmethod
    def can_broadcast(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
        """
        Check if two shapes can be broadcasted together
        Rules:
        1. Start from rightmost (trailing) dimensions
        2. Dimensions are compatible if they are equal OR one is 1
        3. Missing dimensions are treated as 1
        """
        # Pad shorter shape with 1s on the left
        max_ndim = max(len(shape1), len(shape2))
        shape1_padded = (1,) * (max_ndim - len(shape1)) + shape1
        shape2_padded = (1,) * (max_ndim - len(shape2)) + shape2
        
        # Check compatibility dimension by dimension
        for s1, s2 in zip(shape1_padded, shape2_padded):
            if s1 != s2 and s1 != 1 and s2 != 1:
                return False
        
        return True
    
    @staticmethod
    def broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the resulting shape after broadcasting
        """
        if not BroadcastingEngine.can_broadcast(shape1, shape2):
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
        
        # Pad shorter shape with 1s
        max_ndim = max(len(shape1), len(shape2))
        shape1_padded = (1,) * (max_ndim - len(shape1)) + shape1
        shape2_padded = (1,) * (max_ndim - len(shape2)) + shape2
        
        # Compute result shape
        result_shape = []
        for s1, s2 in zip(shape1_padded, shape2_padded):
            result_shape.append(max(s1, s2))
        
        return tuple(result_shape)
    
    @staticmethod
    def compute_broadcast_strides(original_shape: Tuple[int, ...], 
                                broadcast_shape: Tuple[int, ...],
                                original_strides: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute strides for broadcasting without copying data
        Key insight: When a dimension is size 1, set its stride to 0
        """
        # Pad original shape and strides to match broadcast shape
        ndim_diff = len(broadcast_shape) - len(original_shape)
        padded_shape = (1,) * ndim_diff + original_shape
        padded_strides = (0,) * ndim_diff + original_strides
        
        broadcast_strides = []
        for orig_size, broad_size, orig_stride in zip(padded_shape, broadcast_shape, padded_strides):
            if orig_size == 1 and broad_size > 1:
                # Broadcasting dimension: stride = 0 (reuse same element)
                broadcast_strides.append(0)
            else:
                # Normal dimension: keep original stride
                broadcast_strides.append(orig_stride)
        
        return tuple(broadcast_strides)
    
    @staticmethod
    def demonstrate_broadcasting_step_by_step(shape1: Tuple[int, ...], shape2: Tuple[int, ...]):
        """Show detailed broadcasting process"""
        
        print(f"\nBroadcasting Analysis: {shape1} + {shape2}")
        print("-" * 50)
        
        # Step 1: Check compatibility
        compatible = BroadcastingEngine.can_broadcast(shape1, shape2)
        print(f"1. Compatibility check: {'✓' if compatible else '✗'}")
        
        if not compatible:
            print("   Cannot broadcast - incompatible dimensions")
            return
        
        # Step 2: Pad shapes
        max_ndim = max(len(shape1), len(shape2))
        shape1_padded = (1,) * (max_ndim - len(shape1)) + shape1
        shape2_padded = (1,) * (max_ndim - len(shape2)) + shape2
        
        print(f"2. Padded shapes:")
        print(f"   Shape 1: {shape1} -> {shape1_padded}")
        print(f"   Shape 2: {shape2} -> {shape2_padded}")
        
        # Step 3: Dimension-by-dimension analysis
        print(f"3. Dimension analysis:")
        result_shape = []
        for i, (s1, s2) in enumerate(zip(shape1_padded, shape2_padded)):
            result_dim = max(s1, s2)
            result_shape.append(result_dim)
            
            if s1 == s2:
                status = "match"
            elif s1 == 1:
                status = "broadcast shape1"
            elif s2 == 1:
                status = "broadcast shape2"
            else:
                status = "ERROR"
            
            print(f"   Dim {i}: {s1} × {s2} -> {result_dim} ({status})")
        
        print(f"4. Result shape: {tuple(result_shape)}")
        
        # Step 4: Memory layout implications
        print(f"5. Memory implications:")
        if 1 in shape1_padded or 1 in shape2_padded:
            print("   Broadcasting will reuse memory (no data copying)")
            print("   Broadcasted dimensions have stride = 0")
        else:
            print("   No broadcasting needed - shapes already compatible")

# ============================================================================
# PART 3: GRADIENT BROADCASTING (THE TRICKY PART!)
# ============================================================================

class GradientBroadcasting:
    """
    Handle gradient computation through broadcasted operations
    This is one of the most complex parts of automatic differentiation
    """
    
    @staticmethod
    def sum_to_shape(gradient: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Sum gradient tensor to match the target shape
        This reverses the broadcasting operation
        """
        # Start with the gradient
        result = gradient.copy()
        
        print(f"    Reducing gradient: {gradient.shape} -> {target_shape}")
        
        # Handle dimension differences
        ndim_diff = result.ndim - len(target_shape)
        if ndim_diff > 0:
            # Sum over extra leading dimensions
            for _ in range(ndim_diff):
                result = np.sum(result, axis=0)
            print(f"    After removing extra dims: {result.shape}")
        
        # Handle size-1 dimensions that were broadcasted
        for i, (result_size, target_size) in enumerate(zip(result.shape, target_shape)):
            if target_size == 1 and result_size > 1:
                # This dimension was broadcasted, sum it up
                result = np.sum(result, axis=i, keepdims=True)
                print(f"    After summing dim {i}: {result.shape}")
        
        assert result.shape == target_shape, f"Failed to reduce to target shape: {result.shape} != {target_shape}"
        return result
    
    @staticmethod
    def demonstrate_gradient_broadcasting():
        """Show how gradients flow through broadcasted operations"""
        
        print("=" * 60)
        print("GRADIENT BROADCASTING DEMONSTRATION")
        print("=" * 60)
        
        # Example: (3,1) + (4,) -> (3,4)
        # Forward: a + b = c
        # Backward: dc/da and dc/db
        
        print("Forward pass: a(3,1) + b(4,) -> c(3,4)")
        
        a = np.array([[1.0], [2.0], [3.0]])  # Shape (3,1)
        b = np.array([10.0, 20.0, 30.0, 40.0])  # Shape (4,)
        
        print(f"a.shape = {a.shape}, b.shape = {b.shape}")
        
        # Forward pass
        c = a + b  # Broadcasting happens here
        print(f"c.shape = {c.shape}")
        print(f"c =\n{c}")
        
        # Backward pass: assume gradient w.r.t. c is all ones
        grad_c = np.ones_like(c)  # Shape (3,4)
        print(f"\nBackward pass: grad_c.shape = {grad_c.shape}")
        print(f"grad_c =\n{grad_c}")
        
        # Compute gradients
        print(f"\nComputing gradients:")
        
        # da/dc: sum over broadcasted dimensions
        grad_a = GradientBroadcasting.sum_to_shape(grad_c, a.shape)
        print(f"grad_a.shape = {grad_a.shape}")
        print(f"grad_a = {grad_a.flatten()}")
        
        # db/dc: sum over broadcasted dimensions  
        grad_b = GradientBroadcasting.sum_to_shape(grad_c, b.shape)
        print(f"grad_b.shape = {grad_b.shape}")
        print(f"grad_b = {grad_b}")
        
        # Verification: each element of a affects 4 elements of c
        # each element of b affects 3 elements of c
        print(f"\nVerification:")
        print(f"Each a[i] affects 4 elements of c -> grad_a should be [4,4,4]: {grad_a.flatten()}")
        print(f"Each b[j] affects 3 elements of c -> grad_b should be [3,3,3,3]: {grad_b}")

# ============================================================================
# PART 4: CACHE-EFFICIENT OPERATIONS
# ============================================================================

class CacheOptimization:
    """
    Demonstrate cache-efficient tensor operations
    """
    
    @staticmethod
    def naive_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Naive matrix multiplication (cache-unfriendly)"""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        C = np.zeros((M, N), dtype=A.dtype)
        
        # i-j-k order: accesses B column-wise (bad for cache)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    C[i, j] += A[i, k] * B[k, j]
        
        return C
    
    @staticmethod
    def cache_friendly_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Cache-friendly matrix multiplication"""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        C = np.zeros((M, N), dtype=A.dtype)
        
        # i-k-j order: accesses B row-wise (good for cache)
        for i in range(M):
            for k in range(K):
                a_ik = A[i, k]  # Load once, reuse
                for j in range(N):
                    C[i, j] += a_ik * B[k, j]
        
        return C
    
    @staticmethod
    def blocked_matrix_multiply(A: np.ndarray, B: np.ndarray, block_size: int = 64) -> np.ndarray:
        """Blocked matrix multiplication for cache efficiency"""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        C = np.zeros((M, N), dtype=A.dtype)
        
        # Process in blocks that fit in cache
        for ii in range(0, M, block_size):
            for jj in range(0, N, block_size):
                for kk in range(0, K, block_size):
                    # Define block boundaries
                    i_end = min(ii + block_size, M)
                    j_end = min(jj + block_size, N)
                    k_end = min(kk + block_size, K)
                    
                    # Multiply blocks
                    for i in range(ii, i_end):
                        for k in range(kk, k_end):
                            a_ik = A[i, k]
                            for j in range(jj, j_end):
                                C[i, j] += a_ik * B[k, j]
        
        return C
    
    @staticmethod
    def benchmark_matrix_operations():
        """Benchmark different matrix multiplication approaches"""
        
        print("=" * 60)
        print("CACHE OPTIMIZATION BENCHMARK")
        print("=" * 60)
        
        sizes = [128, 256, 512]
        
        for size in sizes:
            print(f"\nTesting {size}x{size} matrices:")
            
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            # NumPy reference (highly optimized)
            start_time = time.time()
            C_numpy = np.dot(A, B)
            numpy_time = time.time() - start_time
            
            # Only test smaller sizes for naive methods (they're too slow)
            if size <= 256:
                # Naive implementation
                start_time = time.time()
                C_naive = CacheOptimization.naive_matrix_multiply(A, B)
                naive_time = time.time() - start_time
                
                # Cache-friendly implementation
                start_time = time.time()
                C_cache = CacheOptimization.cache_friendly_matrix_multiply(A, B)
                cache_time = time.time() - start_time
                
                # Verify correctness
                naive_error = np.max(np.abs(C_naive - C_numpy))
                cache_error = np.max(np.abs(C_cache - C_numpy))
                
                print(f"  NumPy (optimized):    {numpy_time*1000:8.2f}ms")
                print(f"  Naive (i-j-k):       {naive_time*1000:8.2f}ms  (error: {naive_error:.2e})")
                print(f"  Cache-friendly:      {cache_time*1000:8.2f}ms  (error: {cache_error:.2e})")
                print(f"  Naive vs Cache:      {naive_time/cache_time:.1f}x speedup")
                print(f"  NumPy vs Cache:      {cache_time/numpy_time:.1f}x speedup")
            else:
                print(f"  NumPy (optimized):    {numpy_time*1000:8.2f}ms")
                print(f"  (Skipping naive methods - too slow)")

# ============================================================================
# PART 5: REAL-WORLD BROADCASTING EXAMPLES
# ============================================================================

def demonstrate_common_broadcasting_patterns():
    """Show common broadcasting patterns in deep learning"""
    
    print("=" * 60)
    print("COMMON BROADCASTING PATTERNS IN DEEP LEARNING")
    print("=" * 60)
    
    # Pattern 1: Adding bias to linear layer
    print("1. Adding bias to linear layer output:")
    batch_size, features = 32, 128
    linear_output = np.random.randn(batch_size, features)  # (32, 128)
    bias = np.random.randn(features)  # (128,)
    
    print(f"   Linear output: {linear_output.shape}")
    print(f"   Bias: {bias.shape}")
    
    BroadcastingEngine.demonstrate_broadcasting_step_by_step(linear_output.shape, bias.shape)
    
    # Pattern 2: Batch normalization
    print("\n2. Batch normalization:")
    batch_data = np.random.randn(32, 3, 224, 224)  # (N, C, H, W)
    gamma = np.random.randn(3)  # Scale parameter per channel
    beta = np.random.randn(3)   # Shift parameter per channel
    
    print(f"   Batch data: {batch_data.shape}")
    print(f"   Gamma (scale): {gamma.shape}")
    print(f"   Beta (shift): {beta.shape}")
    
    # Need to reshape for broadcasting: (3,) -> (1, 3, 1, 1)
    gamma_reshaped = gamma.reshape(1, 3, 1, 1)
    beta_reshaped = beta.reshape(1, 3, 1, 1)
    
    print(f"   Gamma reshaped: {gamma_reshaped.shape}")
    BroadcastingEngine.demonstrate_broadcasting_step_by_step(batch_data.shape, gamma_reshaped.shape)
    
    # Pattern 3: Attention mechanism
    print("\n3. Attention weights broadcasting:")
    seq_len, d_model = 50, 512
    queries = np.random.randn(seq_len, d_model)  # (50, 512)
    keys = np.random.randn(seq_len, d_model)     # (50, 512)
    
    # Compute attention scores: Q @ K^T
    attention_scores = queries @ keys.T  # (50, 50)
    
    # Apply mask (upper triangular for causal attention)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9  # (50, 50)
    
    print(f"   Attention scores: {attention_scores.shape}")
    print(f"   Mask: {mask.shape}")
    
    BroadcastingEngine.demonstrate_broadcasting_step_by_step(attention_scores.shape, mask.shape)

def run_complete_demo():
    """Run the complete tensor broadcasting demonstration"""
    
    print("TENSOR BROADCASTING AND MEMORY LAYOUT DEEP DIVE")
    print("=" * 80)
    
    # 1. Memory layout fundamentals
    print("\n1. MEMORY LAYOUT FUNDAMENTALS")
    layouts = [
        TensorLayout((3, 4)),
        TensorLayout((2, 3, 4)),
        TensorLayout((1000, 1000)),
    ]
    
    for layout in layouts:
        layout.demonstrate_memory_layout()
        print()
    
    # 2. Memory access patterns
    compare_memory_layouts()
    
    # 3. Broadcasting mechanics
    print("\n" + "=" * 60)
    print("BROADCASTING MECHANICS")
    print("=" * 60)
    
    test_cases = [
        ((3, 4), (4,)),      # Common: matrix + vector
        ((3, 1), (4,)),      # 2D broadcasting
        ((2, 1, 4), (3, 1)), # Complex broadcasting
        ((1,), (3, 4)),      # Scalar broadcasting
        ((3, 4), (3, 4)),    # No broadcasting needed
        ((3, 4), (2, 4)),    # Incompatible (should fail)
    ]
    
    for shape1, shape2 in test_cases:
        try:
            BroadcastingEngine.demonstrate_broadcasting_step_by_step(shape1, shape2)
        except ValueError as e:
            print(f"\nBroadcasting {shape1} + {shape2}: FAILED - {e}")
        print()
    
    # 4. Gradient broadcasting
    GradientBroadcasting.demonstrate_gradient_broadcasting()
    
    # 5. Cache optimization
    CacheOptimization.benchmark_matrix_operations()
    
    # 6. Real-world patterns
    demonstrate_common_broadcasting_patterns()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Memory layout affects performance dramatically (10x+ differences)")
    print("2. Broadcasting is about stride manipulation, not data copying")
    print("3. Gradient broadcasting requires careful dimension reduction")
    print("4. Cache-friendly access patterns are crucial for performance")
    print("5. Understanding these concepts is essential for framework optimization")

if __name__ == "__main__":
    run_complete_demo()