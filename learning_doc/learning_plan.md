# Tensor Operations and Memory Management - Complete Learning Guide

## Table of Contents
1. [Tensor Fundamentals](#tensor-fundamentals)
2. [Memory Layout and Strides](#memory-layout-and-strides)
3. [Broadcasting Mechanics](#broadcasting-mechanics)
4. [High-Performance Tensor Operations](#high-performance-tensor-operations)
5. [Memory Management Strategies](#memory-management-strategies)
6. [Cache Optimization](#cache-optimization)
7. [Practical Exercises](#practical-exercises)

---

## Tensor Fundamentals

### 1. Mathematical Foundation

#### N-Dimensional Arrays
A tensor is a generalization of vectors and matrices:
- **Scalar** (0D): Single number
- **Vector** (1D): Array of numbers
- **Matrix** (2D): 2D array of numbers  
- **Tensor** (ND): N-dimensional array

#### Tensor Properties
```python
class TensorProperties:
    """Core properties every tensor must have"""
    def __init__(self, data, shape, dtype, device):
        self.data = data        # Raw memory buffer
        self.shape = shape      # Dimensions (e.g., (3, 4, 5))
        self.dtype = dtype      # Data type (float32, int64, etc.)
        self.device = device    # CPU, GPU, etc.
        self.ndim = len(shape)  # Number of dimensions
        self.size = np.prod(shape)  # Total number of elements
        self.itemsize = dtype().itemsize  # Bytes per element
        self.nbytes = self.size * self.itemsize  # Total memory usage
        
        # Calculate strides (bytes to skip for each dimension)
        self.strides = self._calculate_strides(shape, self.itemsize)
    
    def _calculate_strides(self, shape, itemsize):
        """Calculate memory strides for row-major (C-style) layout"""
        strides = []
        stride = itemsize
        for dim_size in reversed(shape):
            strides.insert(0, stride)
            stride *= dim_size
        return tuple(strides)
    
    def linear_index(self, indices):
        """Convert N-D indices to linear memory offset"""
        assert len(indices) == self.ndim
        offset = 0
        for i, (index, stride) in enumerate(zip(indices, self.strides)):
            assert 0 <= index < self.shape[i], f"Index {index} out of bounds for dimension {i}"
            offset += index * stride
        return offset
```

#### Index Calculation Deep Dive
```python
def demonstrate_indexing():
    """Show how multi-dimensional indexing works in memory"""
    
    # Example: 3D tensor with shape (2, 3, 4)
    shape = (2, 3, 4)
    itemsize = 4  # 4 bytes per float32
    
    print("Memory Layout for 3D Tensor (2, 3, 4):")
    print("=" * 50)
    
    # Calculate strides
    strides = []
    stride = itemsize
    for dim_size in reversed(shape):
        strides.insert(0, stride)
        stride *= dim_size
    
    print(f"Shape: {shape}")
    print(f"Strides: {strides} bytes")
    print(f"Total memory: {np.prod(shape) * itemsize} bytes")
    
    # Show memory layout
    print("\nMemory addresses for each element:")
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                offset = i * strides[0] + j * strides[1] + k * strides[2]
                print(f"tensor[{i},{j},{k}] -> byte offset {offset}")
```

### 2. Data Types and Precision

#### Numerical Precision Trade-offs
```python
class DataTypeManager:
    """Manage different data types and their properties"""
    
    DTYPES = {
        'float64': {'size': 8, 'range': '±1.8e308', 'precision': '15-17 digits'},
        'float32': {'size': 4, 'range': '±3.4e38', 'precision': '6-9 digits'},
        'float16': {'size': 2, 'range': '±6.5e4', 'precision': '3-4 digits'},
        'bfloat16': {'size': 2, 'range': '±3.4e38', 'precision': '2-3 digits'},
        'int64': {'size': 8, 'range': '±9.2e18', 'precision': 'exact'},
        'int32': {'size': 4, 'range': '±2.1e9', 'precision': 'exact'},
        'int16': {'size': 2, 'range': '±32767', 'precision': 'exact'},
        'int8': {'size': 1, 'range': '±127', 'precision': 'exact'},
        'bool': {'size': 1, 'range': '{0, 1}', 'precision': 'exact'},
    }
    
    @staticmethod
    def memory_usage_comparison(shape, dtypes):
        """Compare memory usage for different data types"""
        size = np.prod(shape)
        
        print(f"Memory usage for tensor shape {shape}:")
        print("-" * 40)
        
        for dtype in dtypes:
            if dtype in DataTypeManager.DTYPES:
                bytes_per_element = DataTypeManager.DTYPES[dtype]['size']
                total_bytes = size * bytes_per_element
                total_mb = total_bytes / (1024 * 1024)
                
                print(f"{dtype:>10}: {total_bytes:>10,} bytes ({total_mb:.2f} MB)")
```

---

## Memory Layout and Strides

### 1. Row-Major vs Column-Major

#### Memory Layout Comparison
```python
class MemoryLayoutDemo:
    """Demonstrate different memory layout strategies"""
    
    @staticmethod
    def compare_layouts():
        """Compare row-major (C-style) vs column-major (Fortran-style)"""
        
        size = (3, 4)
        
        # Row-major (C-style) - default in NumPy
        arr_c = np.arange(12).reshape(size, order='C')
        
        # Column-major (Fortran-style)
        arr_f = np.arange(12).reshape(size, order='F')
        
        print("Matrix content:")
        print(arr_c)
        print()
        
        print("Row-major (C-style) memory layout:")
        print(f"Strides: {arr_c.strides}")
        print("Memory order:", arr_c.flatten(order='C'))
        print()
        
        print("Column-major (Fortran-style) memory layout:")
        print(f"Strides: {arr_f.strides}")
        print("Memory order:", arr_f.flatten(order='F'))
    
    @staticmethod
    def stride_calculation_demo():
        """Show step-by-step stride calculation"""
        
        shapes = [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]
        itemsize = 4  # float32
        
        for shape in shapes:
            print(f"\nShape: {shape}")
            print(f"Dimensions: {len(shape)}")
            
            # Calculate row-major strides
            strides = []
            stride = itemsize
            for dim_size in reversed(shape):
                strides.insert(0, stride)
                stride *= dim_size
            
            print(f"Strides: {strides}")
            print(f"Total memory: {np.prod(shape) * itemsize} bytes")
```

### 2. Views vs Copies

#### Understanding Memory Sharing
```python
class ViewVsCopyDemo:
    """Demonstrate views vs copies in tensor operations"""
    
    @staticmethod
    def basic_views():
        """Show basic view operations"""
        
        original = np.arange(12).reshape(3, 4)
        print("Original array:")
        print(original)
        
        # Slicing creates a view
        slice_view = original[1:3, 1:3]
        print(f"\nSlice view:")
        print(slice_view)
        print(f"Shares memory: {np.shares_memory(original, slice_view)}")
        
        # Reshaping creates a view (if possible)
        reshaped = original.reshape(4, 3)
        print(f"\nReshaped view:")
        print(reshaped)
        print(f"Shares memory: {np.shares_memory(original, reshaped)}")
        
        # Transpose creates a view
        transposed = original.T
        print(f"\nTransposed view:")
        print(transposed)
        print(f"Transposed strides: {transposed.strides}")
        print(f"Shares memory: {np.shares_memory(original, transposed)}")
    
    @staticmethod
    def when_copies_are_created():
        """Show operations that force copies"""
        
        original = np.arange(12).reshape(3, 4)
        
        operations = [
            ("original.T.reshape(4, 3)", lambda x: x.T.reshape(4, 3)),
            ("original[::2, ::2]", lambda x: x[::2, ::2]),
            ("original + 1", lambda x: x + 1),
            ("np.sqrt(original)", lambda x: np.sqrt(x)),
            ("original[original > 5]", lambda x: x[x > 5]),
        ]
        
        print("Operations and memory sharing:")
        print("-" * 40)
        
        for desc, operation in operations:
            try:
                result = operation(original)
                shares_memory = np.shares_memory(original, result)
                print(f"{desc:<25}: {'View' if shares_memory else 'Copy'}")
            except Exception as e:
                print(f"{desc:<25}: Error - {e}")
```

---

## Broadcasting Mechanics

### 1. Broadcasting Rules Deep Dive

#### Step-by-Step Broadcasting Algorithm
```python
class BroadcastingEngine:
    """Detailed broadcasting implementation and explanation"""
    
    @staticmethod
    def check_broadcastable(shape1, shape2, verbose=True):
        """Check if two shapes can be broadcast together"""
        
        if verbose:
            print(f"Checking broadcastability: {shape1} and {shape2}")
        
        # Step 1: Align shapes from the right
        max_ndim = max(len(shape1), len(shape2))
        aligned_shape1 = (1,) * (max_ndim - len(shape1)) + shape1
        aligned_shape2 = (1,) * (max_ndim - len(shape2)) + shape2
        
        if verbose:
            print(f"Aligned shapes: {aligned_shape1} and {aligned_shape2}")
        
        # Step 2: Check dimension compatibility
        result_shape = []
        compatible = True
        
        for i, (s1, s2) in enumerate(zip(aligned_shape1, aligned_shape2)):
            if s1 == s2:
                result_shape.append(s1)
                if verbose:
                    print(f"Dim {i}: {s1} == {s2} ✓")
            elif s1 == 1:
                result_shape.append(s2)
                if verbose:
                    print(f"Dim {i}: {s1} broadcasts to {s2} ✓")
            elif s2 == 1:
                result_shape.append(s1)
                if verbose:
                    print(f"Dim {i}: {s2} broadcasts to {s1} ✓")
            else:
                compatible = False
                if verbose:
                    print(f"Dim {i}: {s1} incompatible with {s2} ✗")
                break
        
        if compatible:
            if verbose:
                print(f"Result shape: {tuple(result_shape)}")
            return True, tuple(result_shape)
        else:
            if verbose:
                print("Shapes are not broadcastable")
            return False, None
    
    @staticmethod
    def broadcasting_examples():
        """Show various broadcasting scenarios"""
        
        examples = [
            ("Scalar + Vector", (), (5,)),
            ("Vector + Vector (same)", (5,), (5,)),
            ("Vector + Matrix", (5,), (3, 5)),
            ("Matrix + Matrix", (3, 5), (3, 5)),
            ("Matrix + Column Vector", (3, 5), (3, 1)),
            ("Matrix + Row Vector", (3, 5), (1, 5)),
            ("3D + 2D", (2, 3, 5), (3, 5)),
            ("3D + 3D with broadcasting", (2, 1, 5), (1, 3, 1)),
            ("Incompatible shapes", (3, 5), (4, 5)),
        ]
        
        print("Broadcasting Examples:")
        print("=" * 60)
        
        for desc, shape1, shape2 in examples:
            print(f"\n{desc}:")
            print(f"  Shape 1: {shape1}")
            print(f"  Shape 2: {shape2}")
            
            try:
                compatible, result = BroadcastingEngine.check_broadcastable(
                    shape1, shape2, verbose=False
                )
                if compatible:
                    print(f"  Result: {result} ✓")
                else:
                    print(f"  Result: Not broadcastable ✗")
            except Exception as e:
                print(f"  Error: {e}")
```

### 2. Gradient Broadcasting

#### Broadcasting in Reverse Mode AD
```python
class GradientBroadcasting:
    """Handle gradient computation through broadcasted operations"""
    
    @staticmethod
    def sum_to_shape(gradient, target_shape, verbose=True):
        """Reduce gradient tensor to match original tensor shape"""
        
        if verbose:
            print(f"Gradient broadcasting: {gradient.shape} -> {target_shape}")
        
        result = gradient.copy()
        
        # Step 1: Handle extra leading dimensions
        ndim_diff = result.ndim - len(target_shape)
        if ndim_diff > 0:
            # Sum over extra leading dimensions
            axes_to_sum = tuple(range(ndim_diff))
            result = np.sum(result, axis=axes_to_sum)
            if verbose:
                print(f"After summing extra dims: {result.shape}")
        
        # Step 2: Handle broadcasted dimensions (size 1 -> size N)
        for i, (result_size, target_size) in enumerate(zip(result.shape, target_shape)):
            if target_size == 1 and result_size > 1:
                # This dimension was broadcasted, sum it up
                result = np.sum(result, axis=i, keepdims=True)
                if verbose:
                    print(f"After summing dim {i}: {result.shape}")
        
        assert result.shape == target_shape, f"Shape mismatch: {result.shape} != {target_shape}"
        return result
    
    @staticmethod
    def broadcast_gradient_demo():
        """Demonstrate gradient broadcasting with examples"""
        
        print("Gradient Broadcasting Examples:")
        print("=" * 50)
        
        # Example 1: Vector + Scalar
        print("\nExample 1: Vector + Scalar")
        a_shape = (5,)
        b_shape = ()  # scalar
        
        # Forward: c = a + b (broadcasts to (5,))
        c_shape = (5,)
        grad_c = np.ones(c_shape)  # Gradient w.r.t. output
        
        print(f"Forward: a{a_shape} + b{b_shape} -> c{c_shape}")
        print(f"grad_c: {grad_c}")
        
        # Backward: gradients for a and b
        grad_a = GradientBroadcasting.sum_to_shape(grad_c, a_shape, verbose=False)
        grad_b = GradientBroadcasting.sum_to_shape(grad_c, b_shape, verbose=False)
        
        print(f"grad_a: {grad_a} (shape: {grad_a.shape})")
        print(f"grad_b: {grad_b} (shape: {grad_b.shape})")
```

---

## High-Performance Tensor Operations

### 1. BLAS Integration

#### Matrix Multiplication Optimization
```python
class BLASIntegration:
    """Demonstrate BLAS integration for high-performance computing"""
    
    @staticmethod
    def demonstrate_gemm_optimization():
        """Show matrix multiplication optimization techniques"""
        
        import time
        
        sizes = [100, 200, 500, 1000]
        
        print("Matrix Multiplication Performance:")
        print("-" * 50)
        print(f"{'Size':<6} {'Time (s)':<10} {'GFLOPS':<10} {'Efficiency':<12}")
        print("-" * 50)
        
        for n in sizes:
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            
            # Time matrix multiplication
            start_time = time.time()
            C = np.dot(A, B)
            elapsed_time = time.time() - start_time
            
            # Calculate performance metrics
            operations = 2 * n**3  # n³ multiplies + n³ adds
            gflops = operations / elapsed_time / 1e9
            
            # Theoretical peak (rough estimate for modern CPU)
            theoretical_peak = 100  # GFLOPS (adjust based on your CPU)
            efficiency = gflops / theoretical_peak * 100
            
            print(f"{n:<6} {elapsed_time:<10.4f} {gflops:<10.2f} {efficiency:<12.1f}%")
```

### 2. Convolution Implementations

#### im2col Transformation
```python
class ConvolutionOptimization:
    """Demonstrate convolution optimization techniques"""
    
    @staticmethod
    def im2col_transformation(input_data, kernel_size, stride=1, padding=0):
        """Convert convolution to matrix multiplication using im2col"""
        
        batch_size, in_channels, input_h, input_w = input_data.shape
        kernel_h, kernel_w = kernel_size
        
        # Calculate output dimensions
        output_h = (input_h + 2*padding - kernel_h) // stride + 1
        output_w = (input_w + 2*padding - kernel_w) // stride + 1
        
        print(f"im2col transformation:")
        print(f"Input shape: {input_data.shape}")
        print(f"Kernel size: {kernel_size}")
        print(f"Output spatial size: ({output_h}, {output_w})")
        
        # Create im2col matrix
        col_height = in_channels * kernel_h * kernel_w
        col_width = batch_size * output_h * output_w
        
        col_matrix = np.zeros((col_height, col_width))
        
        # Fill column matrix
        col_idx = 0
        for b in range(batch_size):
            for oh in range(output_h):
                for ow in range(output_w):
                    # Extract patch
                    patch_start_h = oh * stride
                    patch_start_w = ow * stride
                    
                    # Handle padding
                    patch = np.zeros((in_channels, kernel_h, kernel_w))
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            ih = patch_start_h + kh - padding
                            iw = patch_start_w + kw - padding
                            if 0 <= ih < input_h and 0 <= iw < input_w:
                                patch[:, kh, kw] = input_data[b, :, ih, iw]
                    
                    # Flatten patch and store in column
                    col_matrix[:, col_idx] = patch.flatten()
                    col_idx += 1
        
        return col_matrix
```

---

## Memory Management Strategies

### 1. Custom Memory Allocators

#### Memory Pool Implementation
```python
class MemoryPool:
    """Simple memory pool for tensor allocation"""
    
    def __init__(self, pool_size_mb=100):
        self.pool_size = pool_size_mb * 1024 * 1024  # Convert to bytes
        self.memory_pool = bytearray(self.pool_size)
        self.allocated_blocks = []
        self.free_blocks = [(0, self.pool_size)]  # (offset, size) tuples
        self.total_allocated = 0
        
        print(f"Memory pool initialized: {pool_size_mb}MB")
    
    def allocate(self, size_bytes, alignment=8):
        """Allocate memory from pool"""
        
        # Align size to boundary
        aligned_size = (size_bytes + alignment - 1) // alignment * alignment
        
        # Find suitable free block
        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= aligned_size:
                # Use this block
                allocated_offset = offset
                
                # Update free blocks
                remaining_size = block_size - aligned_size
                if remaining_size > 0:
                    # Update free block
                    self.free_blocks[i] = (offset + aligned_size, remaining_size)
                else:
                    # Remove free block entirely
                    del self.free_blocks[i]
                
                # Track allocation
                self.allocated_blocks.append((allocated_offset, aligned_size))
                self.total_allocated += aligned_size
                
                return allocated_offset
        
        raise MemoryError(f"Cannot allocate {size_bytes} bytes - insufficient memory")
    
    def deallocate(self, offset, size):
        """Deallocate memory back to pool"""
        
        # Remove from allocated blocks
        self.allocated_blocks = [(o, s) for o, s in self.allocated_blocks if o != offset]
        self.total_allocated -= size
        
        # Add to free blocks and merge adjacent blocks
        self.free_blocks.append((offset, size))
        self.free_blocks.sort(key=lambda x: x[0])  # Sort by offset
        
        # Merge adjacent free blocks
        merged_blocks = []
        for current_offset, current_size in self.free_blocks:
            if merged_blocks and merged_blocks[-1][0] + merged_blocks[-1][1] == current_offset:
                # Merge with previous block
                prev_offset, prev_size = merged_blocks[-1]
                merged_blocks[-1] = (prev_offset, prev_size + current_size)
            else:
                merged_blocks.append((current_offset, current_size))
        
        self.free_blocks = merged_blocks
```

### 2. Reference Counting for Tensors

#### Automatic Memory Management
```python
class TensorWithRefCount:
    """Tensor with automatic reference counting"""
    
    def __init__(self, data, shape, dtype=np.float32):
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.ref_count = 1
        self.size = np.prod(shape)
        self.nbytes = self.size * dtype().itemsize
        
        # Track for debugging
        TensorWithRefCount._total_memory += self.nbytes
    
    _total_memory = 0
    
    def add_ref(self):
        """Increment reference count"""
        self.ref_count += 1
        return self
    
    def remove_ref(self):
        """Decrement reference count and cleanup if needed"""
        self.ref_count -= 1
        
        if self.ref_count <= 0:
            self.cleanup()
    
    def cleanup(self):
        """Clean up tensor memory"""
        TensorWithRefCount._total_memory -= self.nbytes
        self.data = None
    
    @classmethod
    def get_total_memory(cls):
        """Get total memory usage across all tensors"""
        return cls._total_memory
```

---

## Cache Optimization

### 1. Cache-Friendly Access Patterns

#### Memory Access Pattern Analysis
```python
class CacheOptimization:
    """Demonstrate cache-friendly programming techniques"""
    
    @staticmethod
    def matrix_traversal_comparison():
        """Compare row-major vs column-major traversal performance"""
        
        import time
        
        size = 2000
        matrix = np.random.randn(size, size).astype(np.float32)
        
        print(f"Matrix traversal comparison ({size}x{size}):")
        print("-" * 40)
        
        # Row-major traversal (cache-friendly for C-order arrays)
        start_time = time.time()
        total = 0.0
        for i in range(size):
            for j in range(size):
                total += matrix[i, j]
        row_major_time = time.time() - start_time
        
        # Column-major traversal (cache-unfriendly for C-order arrays)
        start_time = time.time()
        total = 0.0
        for j in range(size):
            for i in range(size):
                total += matrix[i, j]
        col_major_time = time.time() - start_time
        
        print(f"Row-major traversal:    {row_major_time:.4f}s")
        print(f"Column-major traversal: {col_major_time:.4f}s")
        print(f"Performance ratio:      {col_major_time/row_major_time:.1f}x slower")
```

### 2. Blocking and Tiling Strategies

#### Cache-Optimized Matrix Operations
```python
class TilingOptimization:
    """Implement tiling strategies for cache optimization"""
    
    @staticmethod
    def blocked_matrix_multiply(A, B, block_size=64):
        """Cache-optimized matrix multiplication using blocking"""
        
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Matrix dimensions must match"
        
        C = np.zeros((M, N), dtype=A.dtype)
        
        # Process in blocks
        for ii in range(0, M, block_size):
            for jj in range(0, N, block_size):
                for kk in range(0, K, block_size):
                    # Define block boundaries
                    i_end = min(ii + block_size, M)
                    j_end = min(jj + block_size, N)
                    k_end = min(kk + block_size, K)
                    
                    # Extract blocks
                    A_block = A[ii:i_end, kk:k_end]
                    B_block = B[kk:k_end, jj:j_end]
                    
                    # Multiply blocks and accumulate
                    C[ii:i_end, jj:j_end] += np.dot(A_block, B_block)
        
        return C
    
    @staticmethod
    def optimal_block_size_analysis():
        """Analyze optimal block size based on cache hierarchy"""
        
        # Typical cache sizes (in KB)
        l1_cache = 32    # L1 cache size
        l2_cache = 256   # L2 cache size
        l3_cache = 8192  # L3 cache size
        
        float_size = 4   # bytes per float32
        
        print("Optimal block size analysis:")
        print("-" * 40)
        
        for cache_name, cache_size_kb in [("L1", l1_cache), ("L2", l2_cache), ("L3", l3_cache)]:
            cache_size_bytes = cache_size_kb * 1024
            
            # For matrix multiplication, we need to store 3 blocks: A, B, and C
            # Assuming square blocks: 3 * block_size^2 * sizeof(float) <= cache_size
            max_block_elements = cache_size_bytes // (3 * float_size)
            optimal_block_size = int(np.sqrt(max_block_elements))
            
            print(f"{cache_name} cache ({cache_size_kb}KB):")
            print(f"  Max elements per block: {max_block_elements:,}")
            print(f"  Optimal block size: {optimal_block_size}")
            print(f"  Memory usage: {3 * optimal_block_size**2 * float_size / 1024:.1f}KB")
            print()
```

---

## Practical Exercises

### Exercise 1: Basic Tensor Implementation
**Objective**: Implement a basic tensor class with proper memory management.

```python
# TODO: Complete the Tensor class
class Tensor:
    def __init__(self, data, dtype=np.float32):
        """
        Initialize tensor with automatic shape inference and stride calculation
        """
        # Your implementation here
        pass
    
    def reshape(self, new_shape):
        """
        Reshape tensor (return view if possible, copy otherwise)
        """
        # Your implementation here
        pass
    
    def __getitem__(self, key):
        """
        Implement tensor indexing and slicing
        """
        # Your implementation here
        pass
    
    def __add__(self, other):
        """
        Implement element-wise addition with broadcasting
        """
        # Your implementation here
        pass
    
    def matmul(self, other):
        """
        Implement matrix multiplication
        """
        # Your implementation here
        pass
```

### Exercise 2: Broadcasting Implementation
**Objective**: Implement broadcasting from scratch with detailed error messages.

```python
# TODO: Implement comprehensive broadcasting
class BroadcastEngine:
    @staticmethod
    def broadcast_shapes(shape1, shape2):
        """
        Compute broadcast result shape with detailed error reporting
        """
        # Your implementation here
        pass
    
    @staticmethod
    def broadcast_arrays(arr1, arr2):
        """
        Broadcast arrays without using NumPy's broadcast_to
        """
        # Your implementation here
        pass
    
    @staticmethod
    def element_wise_op(arr1, arr2, operation):
        """
        Perform element-wise operation with broadcasting
        """
        # Your implementation here
        pass
```

### Exercise 3: Memory Pool for Tensors
**Objective**: Implement a memory pool specifically designed for tensor allocation.

```python
# TODO: Implement TensorMemoryPool
class TensorMemoryPool:
    def __init__(self, pool_size_mb=100):
        # Your implementation here
        pass
    
    def allocate_tensor(self, shape, dtype=np.float32):
        """
        Allocate tensor from memory pool
        """
        # Your implementation here
        pass
    
    def deallocate_tensor(self, tensor):
        """
        Return tensor memory to pool
        """
        # Your implementation here
        pass
    
    def defragment(self):
        """
        Defragment memory pool
        """
        # Your implementation here
        pass
```

---

## Assessment Checklist

**Core Understanding:**
- [ ] Understand tensor fundamentals (shape, strides, memory layout)
- [ ] Can explain row-major vs column-major storage
- [ ] Know when operations create views vs copies
- [ ] Understand broadcasting rules and implementation
- [ ] Can calculate memory offsets from multi-dimensional indices

**Performance Optimization:**
- [ ] Understand cache hierarchy effects on performance
- [ ] Can implement cache-friendly algorithms (blocking, tiling)
- [ ] Know memory access pattern optimization techniques
- [ ] Can analyze and optimize memory usage
- [ ] Understand BLAS integration for high performance

**Implementation Skills:**
- [ ] Can implement basic tensor operations from scratch
- [ ] Can design and implement memory management systems
- [ ] Can implement broadcasting without using library functions
- [ ] Can optimize algorithms for specific hardware characteristics
- [ ] Can profile and debug performance issues

**Advanced Topics:**
- [ ] Can implement custom memory allocators
- [ ] Understand garbage collection strategies for tensors
- [ ] Can design efficient gradient broadcasting for AD
- [ ] Know how to optimize for different data types and precisions
- [ ] Can implement tensor operations for both CPU and GPU

**Practical Applications:**
- [ ] Can build tensor library from scratch
- [ ] Can optimize existing tensor operations
- [ ] Can integrate with BLAS/LAPACK libraries
- [ ] Can design memory-efficient neural network operations
- [ ] Can benchmark and compare different implementations

---

## Recommended Learning Sequence

### Phase 1: Fundamentals (2-3 weeks)
1. Study memory layout and stride calculations
2. Implement basic tensor indexing from scratch
3. Understand view vs copy semantics
4. Learn broadcasting rules step by step

### Phase 2: Performance (2-3 weeks)
1. Study cache hierarchy and memory access patterns
2. Implement cache-optimized matrix operations
3. Learn BLAS integration techniques
4. Practice performance profiling and optimization

### Phase 3: Advanced Implementation (3-4 weeks)
1. Build custom memory management system
2. Implement comprehensive broadcasting engine
3. Add support for multiple data types and devices
4. Optimize for production performance

### Phase 4: Integration (2-3 weeks)
1. Integrate with automatic differentiation
2. Add GPU support and optimization
3. Build complete tensor library
4. Performance testing and benchmarking

---

## Additional Resources

### Books
- **"Computer Systems: A Programmer's Perspective"** by Bryant & O'Hallaron - Memory hierarchy and system optimization
- **"Optimizing Software in C++"** by Agner Fog - Performance optimization techniques
- **"High Performance Computing"** by Dongarra et al. - Parallel computing and linear algebra
- **"BLAS and LAPACK User's Guide"** - Linear algebra library documentation

### Research Papers
- **"ATLAS: Automatically Tuned Linear Algebra Software"** - Automatic optimization of BLAS
- **"Cache-Oblivious Algorithms"** by Harald Prokop - Cache-efficient algorithm design
- **"What Every Programmer Should Know About Memory"** by Ulrich Drepper - Memory optimization
- **"Anatomy of High-Performance Matrix Multiplication"** by Goto & Geijn - GEMM optimization

### Online Resources
- **Intel Optimization Reference Manual** - CPU optimization techniques
- **NVIDIA CUDA Best Practices Guide** - GPU optimization
- **NumPy Internals Documentation** - Understanding NumPy implementation
- **PyTorch ATen Library** - Study production tensor library implementation

### Tools and Software
- **Performance Profiling:**
  - Intel VTune Profiler
  - NVIDIA Nsight Systems/Compute
  - Linux perf tools
  - Valgrind (memory debugging)

- **Development:**
  - OpenBLAS (optimized BLAS implementation)
  - Intel MKL (Math Kernel Library)
  - NVIDIA cuBLAS/cuDNN
  - LLVM/Clang for optimization

### Hands-On Projects

#### Beginner Projects
1. **Memory Layout Explorer** - Visualize how different tensor shapes are stored in memory
2. **Broadcasting Calculator** - Tool to compute broadcast shapes and visualize the process
3. **Stride Calculator** - Compute memory strides for different tensor operations
4. **Basic Tensor Class** - Implement fundamental tensor operations

#### Intermediate Projects
1. **Cache-Optimized GEMM** - Implement high-performance matrix multiplication
2. **Memory Pool Allocator** - Build custom memory management for tensors
3. **Broadcasting Engine** - Complete broadcasting implementation with gradient support
4. **Convolution Optimizer** - Implement im2col and other convolution optimizations

#### Advanced Projects
1. **Production Tensor Library** - Build complete NumPy-like library from scratch
2. **GPU Tensor Operations** - Port tensor operations to CUDA/OpenCL
3. **Automatic Optimization** - Build system that automatically tunes operations
4. **Distributed Tensors** - Implement tensors that span multiple devices/nodes

### Learning Tips

#### For Understanding Memory Layout
1. **Visualize Everything** - Draw memory layouts for different tensor shapes
2. **Implement from Scratch** - Don't use library functions until you understand the internals
3. **Profile Everything** - Measure performance of different approaches
4. **Study Assembly** - Look at generated assembly code for critical operations

#### For Performance Optimization
1. **Start Simple** - Optimize the most common operations first
2. **Measure Twice, Optimize Once** - Always profile before optimizing
3. **Understand Your Hardware** - Know your cache sizes and memory bandwidth
4. **Study Production Code** - Read optimized BLAS implementations

#### For Broadcasting
1. **Work Through Examples** - Practice broadcasting by hand with small examples
2. **Implement Gradients** - Understanding gradient broadcasting is crucial for AD
3. **Test Edge Cases** - Broadcasting has many corner cases to handle
4. **Optimize Memory** - Avoid unnecessary copies in broadcast operations

### Common Pitfalls to Avoid

1. **Premature Optimization** - Understand correctness before optimizing
2. **Ignoring Memory Layout** - Memory access patterns dominate performance
3. **Over-Engineering** - Start simple and add complexity gradually
4. **Not Testing Edge Cases** - Broadcasting and memory management have many edge cases
5. **Forgetting Numerical Stability** - Different data types have different precision requirements

### Assessment Milestones

#### Week 2: Basic Understanding
- Can calculate memory offsets for any tensor shape
- Understands difference between views and copies
- Can implement basic tensor indexing

#### Week 4: Broadcasting Mastery
- Can implement broadcasting algorithm from scratch
- Understands gradient broadcasting for automatic differentiation
- Can optimize broadcasted operations

#### Week 8: Performance Optimization
- Can implement cache-optimized matrix multiplication
- Understands memory hierarchy effects on performance
- Can profile and optimize tensor operations

#### Week 12: Production Ready
- Can build complete tensor library
- Understands integration with GPU computing
- Can optimize for real-world workloads

This comprehensive guide provides everything needed to master tensor operations and memory management for building high-performance deep learning frameworks. Focus on hands-on implementation and always verify your understanding with practical projects.