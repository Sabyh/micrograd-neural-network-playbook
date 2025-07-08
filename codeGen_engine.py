"""
TINYGRAD CODEGEN AND ENGINE EXPLAINED
=====================================
Understanding How TinyGrad Achieves High Performance

This file explains the core concepts behind tinygrad's codegen and engine
systems, which transform high-level tensor operations into optimized hardware code.

🎯 LEARNING OBJECTIVES:
1. Understand the role of computational graphs
2. Learn how operation fusion works
3. See how codegen creates optimized kernels
4. Understand different execution engines
5. Learn about memory management and optimization

📚 BACKGROUND:
TinyGrad's architecture: Python Frontend → Graph IR → CodeGen → Engine → Hardware
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum

# ============================================================================
# PART 1: COMPUTATIONAL GRAPH REPRESENTATION
# ============================================================================

print("📚 PART 1: COMPUTATIONAL GRAPH REPRESENTATION")
print("="*60)

print("""
🎯 WHY COMPUTATIONAL GRAPHS?

Instead of executing operations immediately (eager execution),
modern frameworks build a graph of operations first, then optimize
and execute the entire graph.

Benefits:
✅ Operation fusion (combine multiple ops into one kernel)
✅ Memory optimization (reuse buffers, minimize allocations)
✅ Dead code elimination (remove unused computations)
✅ Automatic differentiation (reverse-mode gradients)
✅ Hardware-specific optimization

Example:
    # Eager execution (like our previous lessons)
    a = x + y    # Execute immediately
    b = a * z    # Execute immediately
    
    # Graph execution (like TinyGrad)
    graph = build_graph([
        ('add', [x, y]),    # Node 1
        ('mul', [node1, z]) # Node 2  
    ])
    result = execute_optimized(graph)  # Execute fused kernel
""")

class OpType(Enum):
    """Types of operations in the computational graph."""
    ADD = "add"
    MUL = "mul"
    MATMUL = "matmul"
    RELU = "relu"
    CONV2D = "conv2d"
    RESHAPE = "reshape"
    SUM = "sum"

class TensorShape:
    """Represents tensor shape with utilities."""
    def __init__(self, dims: List[int]):
        self.dims = tuple(dims)
    
    def __repr__(self):
        return f"Shape{self.dims}"
    
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.dims:
            result *= dim
        return result
    
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dims)

class GraphNode:
    """
    A node in the computational graph.
    
    Each node represents an operation with:
    - Operation type (add, mul, conv2d, etc.)
    - Input dependencies (other nodes)
    - Output shape
    - Metadata for optimization
    """
    
    def __init__(self, 
                 op: OpType, 
                 inputs: List['GraphNode'], 
                 shape: TensorShape,
                 name: Optional[str] = None):
        self.op = op
        self.inputs = inputs
        self.shape = shape
        self.name = name or f"{op.value}_{id(self)}"
        
        # Optimization metadata
        self.can_fuse = True  # Can this op be fused with others?
        self.memory_layout = "contiguous"  # Memory layout preference
        self.device_preference = "gpu"  # Preferred execution device
    
    def __repr__(self):
        input_names = [inp.name for inp in self.inputs]
        return f"Node({self.name}: {self.op.value}({input_names}) -> {self.shape})"

class ComputationalGraph:
    """
    Represents the entire computational graph.
    
    This is similar to TinyGrad's internal graph representation.
    The graph tracks all operations and their dependencies.
    """
    
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.inputs: List[GraphNode] = []  # Graph input nodes
        self.outputs: List[GraphNode] = []  # Graph output nodes
    
    def add_input(self, shape: TensorShape, name: str) -> GraphNode:
        """Add an input tensor to the graph."""
        node = GraphNode(OpType.ADD, [], shape, f"input_{name}")  # Placeholder op
        self.inputs.append(node)
        self.nodes.append(node)
        return node
    
    def add_operation(self, op: OpType, inputs: List[GraphNode], 
                     output_shape: TensorShape, name: Optional[str] = None) -> GraphNode:
        """Add an operation node to the graph."""
        node = GraphNode(op, inputs, output_shape, name)
        self.nodes.append(node)
        return node
    
    def set_outputs(self, outputs: List[GraphNode]):
        """Mark nodes as graph outputs."""
        self.outputs = outputs
    
    def topological_sort(self) -> List[GraphNode]:
        """Return nodes in topological order for execution."""
        visited = set()
        result = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for inp in node.inputs:
                visit(inp)
            result.append(node)
        
        for output in self.outputs:
            visit(output)
        
        return result
    
    def print_graph(self):
        """Print the computational graph."""
        print("Computational Graph:")
        print("-" * 40)
        for i, node in enumerate(self.nodes):
            print(f"{i:2}: {node}")

# Example: Build a simple computational graph
print("\n🔍 EXAMPLE: Building a Computational Graph")

# Create graph: result = (x + y) * z
graph = ComputationalGraph()

# Add inputs
x = graph.add_input(TensorShape([1024]), "x")
y = graph.add_input(TensorShape([1024]), "y") 
z = graph.add_input(TensorShape([1024]), "z")

# Add operations
add_node = graph.add_operation(OpType.ADD, [x, y], TensorShape([1024]), "add_xy")
mul_node = graph.add_operation(OpType.MUL, [add_node, z], TensorShape([1024]), "mul_result")

# Set output
graph.set_outputs([mul_node])

# Print the graph
graph.print_graph()

print("\nExecution order:", [node.name for node in graph.topological_sort()])

# ============================================================================
# PART 2: OPERATION FUSION - THE KEY OPTIMIZATION
# ============================================================================

print("\n\n📚 PART 2: OPERATION FUSION")
print("="*60)

print("""
🎯 WHY OPERATION FUSION IS CRITICAL

Without fusion:
    x + y:     Load x, Load y, Add, Store temp1
    temp1 * z: Load temp1, Load z, Mul, Store result
    
    Total: 6 memory operations + 2 compute operations

With fusion:
    fused_add_mul: Load x, Load y, Load z, Add+Mul, Store result
    
    Total: 4 memory operations + 2 compute operations
    
🚀 BENEFITS:
• Fewer memory accesses (main bottleneck on modern hardware)
• Better cache utilization
• Reduced kernel launch overhead (GPU)
• Higher arithmetic intensity

🧮 FUSION PATTERNS:

Element-wise fusion:
    (a + b) * c → fused_add_mul(a, b, c)
    
Reduction fusion:
    sum(relu(x + y)) → fused_add_relu_sum(x, y)
    
Conv + activation fusion:
    relu(conv2d(x, w)) → fused_conv_relu(x, w)
""")

class FusedKernel:
    """
    Represents a fused kernel that combines multiple operations.
    
    This is the core optimization in modern deep learning frameworks.
    Instead of executing operations one by one, we combine them into
    a single, optimized kernel.
    """
    
    def __init__(self, name: str, ops: List[OpType], 
                 inputs: List[GraphNode], output_shape: TensorShape):
        self.name = name
        self.ops = ops  # List of operations to fuse
        self.inputs = inputs
        self.output_shape = output_shape
        self.kernel_code = None  # Generated kernel code
        
    def __repr__(self):
        op_names = [op.value for op in self.ops]
        return f"FusedKernel({self.name}: {' + '.join(op_names)})"

class FusionOptimizer:
    """
    Analyzes the computational graph and identifies fusion opportunities.
    
    This is similar to TinyGrad's optimization passes that transform
    the graph before code generation.
    """
    
    def __init__(self):
        self.fusable_ops = {OpType.ADD, OpType.MUL, OpType.RELU}  # Element-wise ops
    
    def can_fuse(self, node1: GraphNode, node2: GraphNode) -> bool:
        """Check if two nodes can be fused."""
        # Simple fusion rules (real implementations are much more complex)
        return (
            node1.op in self.fusable_ops and 
            node2.op in self.fusable_ops and
            node1.shape.dims == node2.shape.dims and  # Same shape
            len(node2.inputs) == 1 and node2.inputs[0] == node1  # Direct dependency
        )
    
    def fuse_graph(self, graph: ComputationalGraph) -> List[FusedKernel]:
        """
        Analyze graph and create fused kernels.
        
        This is a simplified version of what TinyGrad does internally.
        """
        kernels = []
        nodes_to_process = graph.topological_sort()
        processed = set()
        
        for node in nodes_to_process:
            if node in processed:
                continue
                
            # Try to build a fusion chain starting from this node
            fusion_chain = [node]
            current = node
            
            # Look for nodes that can be fused with current
            for candidate in nodes_to_process:
                if candidate in processed or candidate == current:
                    continue
                    
                if self.can_fuse(current, candidate):
                    fusion_chain.append(candidate)
                    current = candidate
            
            # Create fused kernel if we have multiple operations
            if len(fusion_chain) > 1:
                ops = [n.op for n in fusion_chain]
                inputs = fusion_chain[0].inputs  # Inputs to the first op
                output_shape = fusion_chain[-1].shape  # Output of last op
                
                kernel = FusedKernel(
                    name=f"fused_{'_'.join(op.value for op in ops)}",
                    ops=ops,
                    inputs=inputs,
                    output_shape=output_shape
                )
                kernels.append(kernel)
                
                for n in fusion_chain:
                    processed.add(n)
            else:
                # Single operation - create individual kernel
                kernel = FusedKernel(
                    name=f"single_{node.op.value}",
                    ops=[node.op],
                    inputs=node.inputs,
                    output_shape=node.shape
                )
                kernels.append(kernel)
                processed.add(node)
        
        return kernels

# Example: Fuse the computational graph
print("\n🔍 EXAMPLE: Operation Fusion")

fusion_optimizer = FusionOptimizer()
fused_kernels = fusion_optimizer.fuse_graph(graph)

print("Fused kernels:")
for kernel in fused_kernels:
    print(f"  {kernel}")

print(f"\nOptimization: {len(graph.nodes)} operations → {len(fused_kernels)} kernels")

# ============================================================================
# PART 3: CODE GENERATION - FROM GRAPHS TO HARDWARE CODE
# ============================================================================

print("\n\n📚 PART 3: CODE GENERATION")
print("="*60)

print("""
🎯 THE CODEGEN CHALLENGE

CodeGen must transform high-level operations into efficient hardware code:

Input: Fused kernel (add + mul operations)
Output: Optimized GPU/CPU kernel code

Challenges:
• Memory access patterns (coalesced reads/writes)
• Thread mapping and work distribution  
• Register usage optimization
• Instruction scheduling
• Different hardware architectures (CUDA, OpenCL, CPU SIMD)

🏭 CODEGEN PIPELINE:

1. ANALYZE: Understand data access patterns
2. TEMPLATE: Select appropriate code template
3. GENERATE: Fill template with operation-specific code
4. OPTIMIZE: Apply low-level optimizations
5. COMPILE: Generate machine code
""")

class DeviceTarget(Enum):
    """Target devices for code generation."""
    CPU = "cpu"
    CUDA = "cuda"
    OPENCL = "opencl"
    METAL = "metal"

class CodeTemplate:
    """
    Base class for code generation templates.
    
    Each target device has different templates for generating
    optimized code.
    """
    
    def __init__(self, target: DeviceTarget):
        self.target = target
    
    @abstractmethod
    def generate_kernel(self, kernel: FusedKernel) -> str:
        """Generate optimized code for the fused kernel."""
        pass

class CUDACodeGen(CodeTemplate):
    """
    CUDA code generation for GPU execution.
    
    This generates CUDA C++ code that can be compiled and executed on NVIDIA GPUs.
    Similar to what TinyGrad does for GPU kernels.
    """
    
    def __init__(self):
        super().__init__(DeviceTarget.CUDA)
    
    def generate_kernel(self, kernel: FusedKernel) -> str:
        """Generate CUDA kernel code."""
        
        # Build parameter list
        params = []
        for i, inp in enumerate(kernel.inputs):
            params.append(f"const float* input{i}")
        params.append("float* output")
        params.append("int n")
        
        param_str = ", ".join(params)
        
        # Generate the kernel body
        body = self._generate_kernel_body(kernel)
        
        # Complete CUDA kernel
        cuda_code = f"""
__global__ void {kernel.name}_kernel({param_str}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
{body}
    }}
}}

// Host function to launch kernel
void {kernel.name}_launch(const std::vector<float*>& inputs, float* output, int n) {{
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    {kernel.name}_kernel<<<grid_size, block_size>>>(
        {', '.join(f'inputs[{i}]' for i in range(len(kernel.inputs)))},
        output, n
    );
    
    cudaDeviceSynchronize();  // Wait for completion
}}
"""
        return cuda_code
    
    def _generate_kernel_body(self, kernel: FusedKernel) -> str:
        """Generate the computational body of the kernel."""
        
        if kernel.ops == [OpType.ADD, OpType.MUL]:
            # Fused add + multiply
            return """        float temp = input0[idx] + input1[idx];  // ADD
        output[idx] = temp * input2[idx];         // MUL"""
        
        elif kernel.ops == [OpType.ADD]:
            # Single add
            return """        output[idx] = input0[idx] + input1[idx];  // ADD"""
        
        elif kernel.ops == [OpType.MUL]:
            # Single multiply  
            return """        output[idx] = input0[idx] * input1[idx];  // MUL"""
        
        elif kernel.ops == [OpType.RELU]:
            # ReLU activation
            return """        output[idx] = fmaxf(0.0f, input0[idx]);   // RELU"""
        
        else:
            # Generic fallback
            ops_str = " + ".join(op.value for op in kernel.ops)
            return f"""        // TODO: Implement {ops_str}
        output[idx] = input0[idx];  // Placeholder"""

class CPUCodeGen(CodeTemplate):
    """
    CPU code generation with SIMD optimization.
    
    Generates optimized C++ code with vectorization hints
    for CPU execution.
    """
    
    def __init__(self):
        super().__init__(DeviceTarget.CPU)
    
    def generate_kernel(self, kernel: FusedKernel) -> str:
        """Generate CPU kernel code with SIMD."""
        
        # Build parameter list
        params = []
        for i, inp in enumerate(kernel.inputs):
            params.append(f"const float* __restrict__ input{i}")
        params.append("float* __restrict__ output")
        params.append("int n")
        
        param_str = ", ".join(params)
        
        # Generate vectorized loop
        body = self._generate_vectorized_body(kernel)
        
        cpu_code = f"""
#include <immintrin.h>  // AVX intrinsics

void {kernel.name}_cpu({param_str}) {{
    const int simd_width = 8;  // AVX 256-bit = 8 floats
    const int vectorized_end = (n / simd_width) * simd_width;
    
    // Vectorized loop (process 8 elements at once)
    #pragma omp parallel for
    for (int i = 0; i < vectorized_end; i += simd_width) {{
{body['vectorized']}
    }}
    
    // Handle remaining elements
    for (int i = vectorized_end; i < n; ++i) {{
{body['scalar']}
    }}
}}
"""
        return cpu_code
    
    def _generate_vectorized_body(self, kernel: FusedKernel) -> Dict[str, str]:
        """Generate both vectorized and scalar versions."""
        
        if kernel.ops == [OpType.ADD, OpType.MUL]:
            return {
                'vectorized': """        __m256 v_input0 = _mm256_load_ps(&input0[i]);
        __m256 v_input1 = _mm256_load_ps(&input1[i]);
        __m256 v_input2 = _mm256_load_ps(&input2[i]);
        __m256 v_temp = _mm256_add_ps(v_input0, v_input1);
        __m256 v_result = _mm256_mul_ps(v_temp, v_input2);
        _mm256_store_ps(&output[i], v_result);""",
                
                'scalar': """        float temp = input0[i] + input1[i];
        output[i] = temp * input2[i];"""
            }
        
        elif kernel.ops == [OpType.ADD]:
            return {
                'vectorized': """        __m256 v_input0 = _mm256_load_ps(&input0[i]);
        __m256 v_input1 = _mm256_load_ps(&input1[i]);
        __m256 v_result = _mm256_add_ps(v_input0, v_input1);
        _mm256_store_ps(&output[i], v_result);""",
                
                'scalar': """        output[i] = input0[i] + input1[i];"""
            }
        
        else:
            return {
                'vectorized': """        // TODO: Implement vectorized version""",
                'scalar': """        output[i] = input0[i];  // Placeholder"""
            }

class CodeGenerator:
    """
    Main code generation system.
    
    This coordinates the entire codegen process:
    1. Select appropriate backend
    2. Generate optimized code
    3. Handle compilation
    """
    
    def __init__(self):
        self.backends = {
            DeviceTarget.CUDA: CUDACodeGen(),
            DeviceTarget.CPU: CPUCodeGen(),
        }
    
    def generate_code(self, kernels: List[FusedKernel], 
                     target: DeviceTarget) -> Dict[str, str]:
        """Generate code for all kernels targeting specific device."""
        
        if target not in self.backends:
            raise ValueError(f"Unsupported target: {target}")
        
        backend = self.backends[target]
        generated_code = {}
        
        for kernel in kernels:
            code = backend.generate_kernel(kernel)
            generated_code[kernel.name] = code
        
        return generated_code

# Example: Generate code for our fused kernels
print("\n🔍 EXAMPLE: Code Generation")

codegen = CodeGenerator()

# Generate CUDA code
cuda_code = codegen.generate_code(fused_kernels, DeviceTarget.CUDA)
print("Generated CUDA code:")
for kernel_name, code in cuda_code.items():
    print(f"\n--- {kernel_name} (CUDA) ---")
    print(code[:500] + "..." if len(code) > 500 else code)

# Generate CPU code  
cpu_code = codegen.generate_code(fused_kernels, DeviceTarget.CPU)
print("\nGenerated CPU code:")
for kernel_name, code in cpu_code.items():
    print(f"\n--- {kernel_name} (CPU) ---")
    print(code[:500] + "..." if len(code) > 500 else code)

# ============================================================================
# PART 4: EXECUTION ENGINE - RUNNING THE OPTIMIZED CODE
# ============================================================================

print("\n\n📚 PART 4: EXECUTION ENGINE")
print("="*60)

print("""
🎯 THE ENGINE'S RESPONSIBILITIES

The execution engine is the runtime system that:

1. 📦 MEMORY MANAGEMENT:
   • Allocate tensor storage on target device
   • Track memory usage and lifetime
   • Implement memory pools for efficiency

2. 🚀 KERNEL EXECUTION:
   • Compile generated code to machine code
   • Launch kernels with optimal parameters
   • Handle synchronization and dependencies

3. 🔄 SCHEDULING:
   • Execute kernels in dependency order
   • Overlap computation and memory transfers
   • Load balance across available hardware

4. 🛠️ DEVICE ABSTRACTION:
   • Provide unified interface across hardware
   • Handle device-specific optimizations
   • Manage multi-GPU execution
""")

class Buffer:
    """
    Represents a memory buffer on a specific device.
    
    This abstracts device memory management across different hardware.
    """
    
    def __init__(self, size: int, device: DeviceTarget, dtype=np.float32):
        self.size = size
        self.device = device
        self.dtype = dtype
        self.ptr = None  # Device memory pointer
        self.host_data = None  # Host copy (for debugging)
    
    def allocate(self):
        """Allocate memory on the target device."""
        if self.device == DeviceTarget.CPU:
            self.ptr = np.zeros(self.size, dtype=self.dtype)
        else:
            # For GPU, this would call cudaMalloc, etc.
            self.ptr = f"gpu_ptr_{id(self)}"  # Placeholder
        
        print(f"Allocated {self.size} elements on {self.device.value}")
    
    def free(self):
        """Free device memory."""
        if self.ptr is not None:
            print(f"Freed buffer on {self.device.value}")
            self.ptr = None

class CompiledKernel:
    """
    Represents a compiled kernel ready for execution.
    
    This contains the compiled machine code and metadata
    needed for efficient execution.
    """
    
    def __init__(self, name: str, target: DeviceTarget, source_code: str):
        self.name = name
        self.target = target
        self.source_code = source_code
        self.compiled_binary = None
        self.launch_params = {}
    
    def compile(self):
        """Compile source code to executable binary."""
        print(f"Compiling {self.name} for {self.target.value}...")
        
        if self.target == DeviceTarget.CUDA:
            # In reality, this would call nvcc or use CUDA runtime compilation
            self.compiled_binary = f"cuda_binary_{self.name}"
            self.launch_params = {
                'block_size': 256,
                'grid_size_formula': 'lambda n: (n + 255) // 256'
            }
        elif self.target == DeviceTarget.CPU:
            # In reality, this would call gcc/clang
            self.compiled_binary = f"cpu_binary_{self.name}"
            self.launch_params = {
                'num_threads': 8  # OpenMP threads
            }
        
        print(f"  Compilation successful: {self.compiled_binary}")
    
    def execute(self, inputs: List[Buffer], output: Buffer):
        """Execute the compiled kernel."""
        print(f"Executing {self.name} on {self.target.value}")
        print(f"  Inputs: {len(inputs)} buffers")
        print(f"  Output: {output.size} elements")
        
        if self.target == DeviceTarget.CUDA:
            grid_size = eval(self.launch_params['grid_size_formula'])(output.size)
            print(f"  CUDA launch: grid={grid_size}, block={self.launch_params['block_size']}")
        elif self.target == DeviceTarget.CPU:
            print(f"  CPU execution: {self.launch_params['num_threads']} threads")

class ExecutionEngine:
    """
    Main execution engine that coordinates the entire execution process.
    
    This is the heart of the runtime system - similar to TinyGrad's
    execution engine that handles device management, compilation,
    and kernel execution.
    """
    
    def __init__(self, target: DeviceTarget):
        self.target = target
        self.compiled_kernels: Dict[str, CompiledKernel] = {}
        self.memory_pool: List[Buffer] = []
        self.allocated_buffers: Dict[str, Buffer] = {}
    
    def compile_kernels(self, generated_code: Dict[str, str]):
        """Compile all generated kernels."""
        print(f"\n🔧 Compiling kernels for {self.target.value}...")
        
        for kernel_name, source_code in generated_code.items():
            kernel = CompiledKernel(kernel_name, self.target, source_code)
            kernel.compile()
            self.compiled_kernels[kernel_name] = kernel
        
        print(f"Compiled {len(self.compiled_kernels)} kernels")
    
    def allocate_buffer(self, name: str, size: int) -> Buffer:
        """Allocate a named buffer."""
        buffer = Buffer(size, self.target)
        buffer.allocate()
        self.allocated_buffers[name] = buffer
        return buffer
    
    def execute_kernel(self, kernel_name: str, input_names: List[str], output_name: str):
        """Execute a specific kernel."""
        if kernel_name not in self.compiled_kernels:
            raise ValueError(f"Kernel {kernel_name} not compiled")
        
        # Get buffers
        input_buffers = [self.allocated_buffers[name] for name in input_names]
        output_buffer = self.allocated_buffers[output_name]
        
        # Execute
        kernel = self.compiled_kernels[kernel_name]
        kernel.execute(input_buffers, output_buffer)
    
    def execute_graph(self, kernels: List[FusedKernel], 
                     input_data: Dict[str, np.ndarray]) -> Dict[str, Buffer]:
        """Execute entire computational graph."""
        print(f"\n🚀 Executing computational graph on {self.target.value}")
        
        # Allocate input buffers and copy data
        for name, data in input_data.items():
            buffer = self.allocate_buffer(name, data.size)
            # In reality, this would copy data to device
            print(f"Copied {name}: {data.shape} to device")
        
        # Execute kernels in dependency order
        results = {}
        for i, kernel in enumerate(kernels):
            # Allocate output buffer
            output_name = f"output_{kernel.name}"
            output_size = kernel.output_shape.numel()
            output_buffer = self.allocate_buffer(output_name, output_size)
            
            # Execute kernel (simplified - real execution would be more complex)
            input_names = [f"input_{inp.name}" for inp in kernel.inputs]
            self.execute_kernel(kernel.name, input_names, output_name)
            
            results[kernel.name] = output_buffer
        
        return results
    
    def cleanup(self):
        """Free all allocated resources."""
        print(f"\n🧹 Cleaning up {len(self.allocated_buffers)} buffers...")
        for buffer in self.allocated_buffers.values():
            buffer.free()
        self.allocated_buffers.clear()

# Example: Complete end-to-end execution
print("\n🔍 EXAMPLE: Complete Execution Pipeline")

# Create execution engines for different targets
cuda_engine = ExecutionEngine(DeviceTarget.CUDA)
cpu_engine = ExecutionEngine(DeviceTarget.CPU)

# Compile kernels
cuda_engine.compile_kernels(cuda_code)
cpu_engine.compile_kernels(cpu_code)

# Prepare input data
input_data = {
    "x": np.random.randn(1024).astype(np.float32),
    "y": np.random.randn(1024).astype(np.float32),
    "z": np.random.randn(1024).astype(np.float32)
}

print(f"Input data shapes: {[(name, data.shape) for name, data in input_data.items()]}")

# Execute on CUDA
print("\n" + "="*50)
print("CUDA EXECUTION")
print("="*50)
cuda_results = cuda_engine.execute_graph(fused_kernels, input_data)
cuda_engine.cleanup()

# Execute on CPU
print("\n" + "="*50)
print("CPU EXECUTION") 
print("="*50)
cpu_results = cpu_engine.execute_graph(fused_kernels, input_data)
cpu_engine.cleanup()

# ============================================================================
# PART 5: MEMORY OPTIMIZATION - THE HIDDEN COMPLEXITY
# ============================================================================

print("\n\n📚 PART 5: MEMORY OPTIMIZATION")
print("="*60)

print("""
🎯 MEMORY IS THE BOTTLENECK

Modern AI models are memory-bound, not compute-bound:
• GPT-3: 175B parameters × 4 bytes = 700GB memory needed
• GPU memory: 40-80GB typical
• Memory bandwidth: 1-2 TB/s
• Compute throughput: 300+ TFlops/s

The challenge: Keep the compute units fed with data!

🧮 MEMORY OPTIMIZATION TECHNIQUES:

1. BUFFER REUSE:
   temp1 = a + b
   temp2 = temp1 * c  # Can reuse temp1's buffer for temp2!
   
2. IN-PLACE OPERATIONS:
   a = a + b  # Overwrite a's buffer directly
   
3. MEMORY POOLING:
   Pre-allocate large chunks, sub-allocate as needed
   
4. GRADIENT CHECKPOINTING:
   Trade compute for memory by recomputing activations
   
5. MEMORY MAPPING:
   Stream data from disk/CPU during computation
""")

class MemoryPool:
    """
    Memory pool for efficient buffer management.
    
    Instead of allocating/freeing individual buffers,
    maintain a pool of reusable memory chunks.
    """
    
    def __init__(self, device: DeviceTarget, pool_size: int = 1024*1024*1024):  # 1GB
        self.device = device
        self.pool_size = pool_size
        self.free_chunks: List[Tuple[int, int]] = [(0, pool_size)]  # (offset, size)
        self.allocated_chunks: Dict[int, Tuple[int, int]] = {}  # id -> (offset, size)
        self.next_id = 0
        
        print(f"Created memory pool: {pool_size // (1024*1024)}MB on {device.value}")
    
    def allocate(self, size: int) -> Optional[int]:
        """Allocate a chunk from the pool. Returns chunk ID or None if failed."""
        # Find first fit
        for i, (offset, chunk_size) in enumerate(self.free_chunks):
            if chunk_size >= size:
                # Allocate from this chunk
                chunk_id = self.next_id
                self.next_id += 1
                
                self.allocated_chunks[chunk_id] = (offset, size)
                
                # Update free list
                if chunk_size == size:
                    # Exact fit - remove chunk
                    del self.free_chunks[i]
                else:
                    # Partial fit - shrink chunk
                    self.free_chunks[i] = (offset + size, chunk_size - size)
                
                print(f"Allocated {size} bytes (ID: {chunk_id}) at offset {offset}")
                return chunk_id
        
        print(f"Failed to allocate {size} bytes - out of memory")
        return None
    
    def free(self, chunk_id: int):
        """Free a previously allocated chunk."""
        if chunk_id not in self.allocated_chunks:
            return
        
        offset, size = self.allocated_chunks[chunk_id]
        del self.allocated_chunks[chunk_id]
        
        # Add back to free list (simplified - no coalescing)
        self.free_chunks.append((offset, size))
        self.free_chunks.sort()  # Keep sorted by offset
        
        print(f"Freed chunk {chunk_id}: {size} bytes at offset {offset}")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        total_allocated = sum(size for _, size in self.allocated_chunks.values())
        total_free = sum(size for _, size in self.free_chunks)
        
        return {
            'total_pool': self.pool_size,
            'allocated': total_allocated,
            'free': total_free,
            'utilization': total_allocated / self.pool_size
        }

class LivenessAnalyzer:
    """
    Analyzes tensor lifetimes to optimize memory reuse.
    
    This determines when each tensor is created and when it's
    last used, enabling optimal buffer reuse.
    """
    
    def __init__(self):
        self.tensor_lifetimes: Dict[str, Tuple[int, int]] = {}  # name -> (birth, death)
    
    def analyze_graph(self, graph: ComputationalGraph) -> Dict[str, Tuple[int, int]]:
        """Analyze tensor lifetimes in the computational graph."""
        
        execution_order = graph.topological_sort()
        lifetimes = {}
        
        # Track when each tensor is created (birth)
        for step, node in enumerate(execution_order):
            lifetimes[node.name] = [step, step]  # [birth, death]
        
        # Track when each tensor is last used (death)
        for step, node in enumerate(execution_order):
            for input_node in node.inputs:
                if input_node.name in lifetimes:
                    lifetimes[input_node.name][1] = step  # Update death time
        
        # Convert to tuples
        self.tensor_lifetimes = {name: tuple(times) for name, times in lifetimes.items()}
        
        return self.tensor_lifetimes
    
    def can_reuse_buffer(self, tensor1: str, tensor2: str) -> bool:
        """Check if tensor2 can reuse tensor1's buffer."""
        if tensor1 not in self.tensor_lifetimes or tensor2 not in self.tensor_lifetimes:
            return False
        
        t1_birth, t1_death = self.tensor_lifetimes[tensor1]
        t2_birth, t2_death = self.tensor_lifetimes[tensor2]
        
        # tensor2 can reuse tensor1's buffer if tensor1 dies before tensor2 is born
        return t1_death < t2_birth
    
    def print_lifetimes(self):
        """Print tensor lifetime analysis."""
        print("Tensor Lifetimes:")
        print("-" * 40)
        for name, (birth, death) in sorted(self.tensor_lifetimes.items()):
            print(f"{name:15}: steps {birth:2} -> {death:2} (lifetime: {death-birth+1})")

class MemoryOptimizer:
    """
    Optimizes memory allocation using liveness analysis.
    
    This implements the algorithms that TinyGrad uses to minimize
    memory usage through intelligent buffer reuse.
    """
    
    def __init__(self, memory_pool: MemoryPool):
        self.memory_pool = memory_pool
        self.liveness_analyzer = LivenessAnalyzer()
        self.buffer_assignments: Dict[str, int] = {}  # tensor -> buffer_id
    
    def optimize_graph(self, graph: ComputationalGraph) -> Dict[str, int]:
        """Optimize memory allocation for the entire graph."""
        
        print("\n🧠 Memory Optimization Analysis")
        print("-" * 40)
        
        # Analyze tensor lifetimes
        lifetimes = self.liveness_analyzer.analyze_graph(graph)
        self.liveness_analyzer.print_lifetimes()
        
        # Simple greedy allocation with reuse
        execution_order = graph.topological_sort()
        active_tensors = set()
        buffer_pool = []  # Available buffers for reuse
        
        for step, node in enumerate(execution_order):
            tensor_name = node.name
            tensor_size = node.shape.numel()
            
            # Remove dead tensors from active set
            dead_tensors = []
            for active_tensor in active_tensors:
                _, death_time = lifetimes[active_tensor]
                if death_time < step:
                    dead_tensors.append(active_tensor)
            
            for dead_tensor in dead_tensors:
                active_tensors.remove(dead_tensor)
                # Return buffer to pool for reuse
                if dead_tensor in self.buffer_assignments:
                    buffer_id = self.buffer_assignments[dead_tensor]
                    buffer_pool.append(buffer_id)
                    print(f"  Buffer {buffer_id} freed from {dead_tensor}, available for reuse")
            
            # Allocate buffer for current tensor
            if buffer_pool:
                # Reuse existing buffer
                buffer_id = buffer_pool.pop(0)
                self.buffer_assignments[tensor_name] = buffer_id
                print(f"  Reused buffer {buffer_id} for {tensor_name}")
            else:
                # Allocate new buffer
                buffer_id = self.memory_pool.allocate(tensor_size * 4)  # 4 bytes per float
                if buffer_id is not None:
                    self.buffer_assignments[tensor_name] = buffer_id
                    print(f"  Allocated new buffer {buffer_id} for {tensor_name}")
                else:
                    print(f"  ERROR: Out of memory for {tensor_name}")
            
            active_tensors.add(tensor_name)
        
        # Print final memory usage
        usage = self.memory_pool.get_memory_usage()
        print(f"\nMemory Usage:")
        print(f"  Total pool: {usage['total_pool'] // (1024*1024)}MB")
        print(f"  Allocated: {usage['allocated'] // (1024*1024)}MB")
        print(f"  Utilization: {usage['utilization']*100:.1f}%")
        
        return self.buffer_assignments

# Example: Memory optimization
print("\n🔍 EXAMPLE: Memory Optimization")

# Create memory pool and optimizer
memory_pool = MemoryPool(DeviceTarget.CUDA, pool_size=64*1024*1024)  # 64MB
optimizer = MemoryOptimizer(memory_pool)

# Optimize memory for our graph
buffer_assignments = optimizer.optimize_graph(graph)

print(f"\nBuffer assignments: {buffer_assignments}")

# ============================================================================
# PART 6: PUTTING IT ALL TOGETHER - THE COMPLETE PIPELINE
# ============================================================================

print("\n\n📚 PART 6: THE COMPLETE TINYGRAD-STYLE PIPELINE")
print("="*60)

print("""
🎯 THE FULL PIPELINE

Now let's see how all components work together in a TinyGrad-style framework:

1. 📊 BUILD GRAPH: Convert high-level operations to computational graph
2. 🔧 OPTIMIZE: Apply fusion and other graph optimizations  
3. 🏭 CODEGEN: Generate optimized kernels for target hardware
4. 💾 MEMORY: Analyze lifetimes and optimize memory allocation
5. 🚀 EXECUTE: Compile and run optimized kernels
6. 🧹 CLEANUP: Free resources and prepare for next iteration

This is exactly what happens inside TinyGrad when you write:
    result = (x + y) * z
""")

class TinyGradStyle:
    """
    A simplified TinyGrad-style deep learning framework.
    
    This demonstrates how all the components work together:
    graph construction, optimization, codegen, and execution.
    """
    
    def __init__(self, device: DeviceTarget = DeviceTarget.CUDA):
        self.device = device
        self.current_graph = ComputationalGraph()
        self.fusion_optimizer = FusionOptimizer()
        self.codegen = CodeGenerator()
        self.memory_pool = MemoryPool(device)
        self.execution_engine = ExecutionEngine(device)
        
        print(f"🚀 TinyGrad-style framework initialized for {device.value}")
    
    def tensor(self, data: np.ndarray, name: str) -> GraphNode:
        """Create a tensor (input to the graph)."""
        shape = TensorShape(list(data.shape))
        node = self.current_graph.add_input(shape, name)
        print(f"Created tensor '{name}' with shape {shape}")
        return node
    
    def add(self, a: GraphNode, b: GraphNode, name: str = None) -> GraphNode:
        """Add two tensors."""
        if name is None:
            name = f"add_{len(self.current_graph.nodes)}"
        
        # Assume same shape for simplicity
        output_shape = a.shape
        node = self.current_graph.add_operation(OpType.ADD, [a, b], output_shape, name)
        print(f"Added operation: {name} = {a.name} + {b.name}")
        return node
    
    def mul(self, a: GraphNode, b: GraphNode, name: str = None) -> GraphNode:
        """Multiply two tensors."""
        if name is None:
            name = f"mul_{len(self.current_graph.nodes)}"
        
        output_shape = a.shape
        node = self.current_graph.add_operation(OpType.MUL, [a, b], output_shape, name)
        print(f"Added operation: {name} = {a.name} * {b.name}")
        return node
    
    def compute(self, output: GraphNode, input_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Execute the complete pipeline to compute the result.
        
        This is the main entry point that orchestrates everything.
        """
        print(f"\n🎯 Computing {output.name} using TinyGrad-style pipeline")
        print("="*60)
        
        # Step 1: Finalize graph
        self.current_graph.set_outputs([output])
        print(f"\n1️⃣ Graph construction complete")
        self.current_graph.print_graph()
        
        # Step 2: Optimization (fusion)
        print(f"\n2️⃣ Applying optimizations...")
        fused_kernels = self.fusion_optimizer.fuse_graph(self.current_graph)
        print(f"Optimization result: {len(self.current_graph.nodes)} ops → {len(fused_kernels)} kernels")
        for kernel in fused_kernels:
            print(f"  {kernel}")
        
        # Step 3: Code generation
        print(f"\n3️⃣ Generating optimized code...")
        generated_code = self.codegen.generate_code(fused_kernels, self.device)
        print(f"Generated {len(generated_code)} kernel(s) for {self.device.value}")
        
        # Step 4: Memory optimization
        print(f"\n4️⃣ Optimizing memory allocation...")
        memory_optimizer = MemoryOptimizer(self.memory_pool)
        buffer_assignments = memory_optimizer.optimize_graph(self.current_graph)
        
        # Step 5: Compilation and execution
        print(f"\n5️⃣ Compiling and executing...")
        self.execution_engine.compile_kernels(generated_code)
        results = self.execution_engine.execute_graph(fused_kernels, input_data)
        
        # Step 6: Cleanup
        print(f"\n6️⃣ Cleaning up...")
        self.execution_engine.cleanup()
        
        print(f"\n✅ Computation complete!")
        return results
    
    def reset(self):
        """Reset for next computation."""
        self.current_graph = ComputationalGraph()
        print("🔄 Framework reset for next computation")

# Example: Complete TinyGrad-style computation
print("\n🔍 EXAMPLE: Complete TinyGrad-Style Pipeline")

# Create framework
framework = TinyGradStyle(DeviceTarget.CUDA)

# Build computation: result = (x + y) * z
print("\n📊 Building computation graph...")
x = framework.tensor(np.random.randn(1024), "x")
y = framework.tensor(np.random.randn(1024), "y")
z = framework.tensor(np.random.randn(1024), "z")

temp = framework.add(x, y, "add_xy")
result = framework.mul(temp, z, "final_result")

# Execute the complete pipeline
input_data = {
    "x": np.random.randn(1024).astype(np.float32),
    "y": np.random.randn(1024).astype(np.float32),
    "z": np.random.randn(1024).astype(np.float32)
}

final_results = framework.compute(result, input_data)

print(f"\nFinal results: {list(final_results.keys())}")

# ============================================================================
# PART 7: KEY INSIGHTS AND COMPARISON TO REAL TINYGRAD
# ============================================================================

print("\n\n📚 PART 7: KEY INSIGHTS")
print("="*60)

print("""
🎯 WHAT YOU'VE LEARNED ABOUT TINYGRAD'S ARCHITECTURE:

1. 📊 COMPUTATIONAL GRAPHS:
   • High-level operations become nodes in a graph
   • Dependencies tracked automatically
   • Enables global optimization across operations

2. 🔧 OPERATION FUSION:
   • Multiple operations combined into single kernels
   • Dramatically reduces memory bandwidth requirements
   • Key optimization for modern hardware performance

3. 🏭 CODE GENERATION:
   • Graph operations transformed to hardware-specific code
   • Templates enable targeting different devices (CUDA, CPU, etc.)
   • Generated code is highly optimized for the specific computation

4. 💾 MEMORY OPTIMIZATION:
   • Liveness analysis determines when tensors can be freed
   • Buffer reuse minimizes memory allocation overhead
   • Critical for large models that exceed GPU memory

5. 🚀 EXECUTION ENGINE:
   • Coordinates compilation, memory management, and execution
   • Handles device-specific details and optimization
   • Provides unified interface across different hardware

🔥 HOW THIS COMPARES TO REAL TINYGRAD:

Our Simplified Version:
✅ Core concepts and architecture
✅ Basic fusion and optimization
✅ Code generation principles
✅ Memory management fundamentals

Real TinyGrad Additions:
🚀 Advanced fusion patterns (reduce, reshape, etc.)
🚀 Lazy evaluation with shape tracking
🚀 Multi-device and distributed execution
🚀 Automatic differentiation integration
🚀 Advanced memory optimizations (gradient checkpointing)
🚀 JIT compilation with caching
🚀 Shape inference and broadcasting
🚀 Custom CUDA kernel generation

🌟 WHY TINYGRAD'S APPROACH IS REVOLUTIONARY:

1. SIMPLICITY: Much smaller codebase than PyTorch/TensorFlow
2. PERFORMANCE: Competitive speed through aggressive optimization
3. TRANSPARENCY: Easy to understand and modify
4. FLEXIBILITY: Easy to add new backends and optimizations
5. RESEARCH-FRIENDLY: Perfect for experimenting with new ideas

🧠 KEY INSIGHTS FOR YOUR OWN FRAMEWORK:

1. 📊 GRAPH-BASED DESIGN:
   Start with eager execution, then add graph compilation
   for optimization opportunities

2. 🔧 FUSION IS CRITICAL:
   Memory bandwidth is the bottleneck, not compute
   Fuse operations aggressively

3. 🏭 TEMPLATE-BASED CODEGEN:
   Use templates for different operation patterns
   Much easier than generating code from scratch

4. 💾 MEMORY IS KING:
   Optimize for memory usage first, then speed
   Buffer reuse can save massive amounts of memory

5. 🚀 START SIMPLE:
   Build basic functionality first, optimize later
   TinyGrad started small and grew incrementally

🎯 NEXT STEPS TO BUILD YOUR OWN TINYGRAD:

1. 🎬 ADD LAZY EVALUATION:
   Don't execute operations immediately
   Build graph and optimize before execution

2. 🔄 IMPLEMENT AUTOGRAD:
   Add automatic differentiation to your graph system
   Track gradients through the computation graph

3. 🎮 ADD MORE BACKENDS:
   Implement CPU, Metal, OpenCL code generation
   Each backend teaches you something new

4. 🧮 ADVANCED FUSION:
   Add support for more complex fusion patterns
   Reduce operations, reshape fusion, etc.

5. 📊 SHAPE INFERENCE:
   Automatically compute output shapes
   Handle broadcasting and dimension changes

Your framework from previous lessons + these concepts = Your own TinyGrad! 🚀

🎊 CONGRATULATIONS!

You now understand the core architecture that powers modern deep learning frameworks!

The concepts you've learned here are used in:
• TinyGrad (obviously!)
• JAX/XLA compilation pipeline
• PyTorch 2.0's torch.compile
• TensorFlow's XLA backend
• MLX (Apple's framework)

This knowledge puts you in the elite group who truly understand
how modern AI frameworks achieve their performance! 🎉
""")

print("\n" + "="*80)
print("🎉 TINYGRAD CODEGEN & ENGINE EXPLANATION COMPLETE!")
print("You now understand the magic behind high-performance deep learning!")
print("="*80)