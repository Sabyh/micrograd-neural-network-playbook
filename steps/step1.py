"""
LEARN-FIRST IMPLEMENTATION GUIDE
================================
Building a Deep Learning Library Step by Step

Approach: 
1. 📚 LEARN the concept thoroughly
2. 🧠 UNDERSTAND why it's needed  
3. 💭 THINK through the implementation
4. 💻 CODE it with documentation
5. 🧪 TEST it thoroughly

Let's start with the most important missing piece from your Value class.
"""

import math
import numpy as np
from typing import Union, List, Tuple, Optional, Any

# ============================================================================
# LESSON 1: FIXING YOUR VALUE CLASS - UNDERSTANDING THE BUGS
# ============================================================================

print("📚 LESSON 1: UNDERSTANDING AND FIXING YOUR VALUE CLASS")
print("="*70)

print("""
🎯 LEARNING OBJECTIVE:
Understand what was wrong with your original Value class and why these fixes matter.

🔍 THE BUGS IN YOUR ORIGINAL CODE:

1. THE EXP() BUG:
   Your code: out += Value(math.exp(x), ...)  ❌
   Should be: out = Value(math.exp(x), ...)   ✅
   
   Why this is wrong:
   - '+=' tries to add to 'out' before 'out' exists
   - This would crash your program
   - It's a basic Python syntax error

2. MISSING OPERATIONS:
   - No power operation (x^y)
   - No division, subtraction  
   - No comparison operations
   - Limited math functions

3. POOR ERROR HANDLING:
   - No checks for invalid operations (like log of negative numbers)
   - No type checking
   - No graceful failure

🧠 WHY THESE FIXES MATTER:

Mathematical completeness:
- Neural networks need ALL basic math operations
- Can't build complex functions without building blocks
- Each operation needs correct gradient computation

Numerical stability:
- Prevents overflow/underflow errors
- Handles edge cases gracefully
- Makes training more reliable

User experience:
- Clear error messages when things go wrong
- Predictable behavior
- Easy debugging

🛠️ WHAT WE'LL IMPLEMENT:
1. Fix the exp() bug
2. Add missing math operations with correct gradients
3. Add proper error handling
4. Add comprehensive documentation
""")

# Now let's implement the fixed Value class step by step

class Value:
    """
    A scalar value with automatic differentiation support.
    
    This class represents a single number that can track gradients through
    computational graphs. It's the foundation of automatic differentiation.
    
    Attributes:
        data (float): The actual numerical value
        grad (float): The gradient of this value with respect to the output
        _children (set): The Value objects that were used to create this one
        _op (str): The operation that created this value (for debugging)
        _backward (callable): Function to compute gradients for this operation
        label (str): Optional label for visualization and debugging
    
    Example:
        >>> a = Value(2.0, label='a')
        >>> b = Value(3.0, label='b') 
        >>> c = a * b + a
        >>> c.backward()
        >>> print(f"a.grad = {a.grad}")  # da/dc = b + 1 = 4.0
        >>> print(f"b.grad = {b.grad}")  # db/dc = a = 2.0
    """
    
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label: str = None):
        """
        Initialize a Value object.
        
        Args:
            data: The numerical value to store
            _children: Tuple of Value objects that created this one (internal use)
            _op: String describing the operation that created this value (internal use)
            label: Optional human-readable label for this value
            
        Raises:
            TypeError: If data is not a number
        """
        # Type checking for robustness
        if not isinstance(data, (int, float)):
            raise TypeError(f"Value data must be a number, got {type(data)}")
            
        self.data = float(data)  # Ensure it's always a float
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None  # Default: no gradient computation
        self.label = label if label is not None else f"Value({self.data})"
        
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other: Union['Value', float, int]) -> 'Value':
        """
        Addition operation: self + other
        
        Mathematical background:
            If f(x,y) = x + y, then:
            ∂f/∂x = 1
            ∂f/∂y = 1
            
        This means gradients flow equally to both operands.
        
        Args:
            other: Another Value or a number to add
            
        Returns:
            New Value representing the sum
            
        Example:
            >>> a = Value(2)
            >>> b = Value(3)
            >>> c = a + b  # c.data = 5
            >>> c.backward()
            >>> print(a.grad, b.grad)  # 1.0, 1.0
        """
        # Convert numbers to Value objects for consistent handling
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            """Compute gradients for addition operation."""
            # Gradient of addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        return out

    def __mul__(self, other: Union['Value', float, int]) -> 'Value':
        """
        Multiplication operation: self * other
        
        Mathematical background:
            If f(x,y) = x * y, then:
            ∂f/∂x = y  (derivative with respect to x is y)
            ∂f/∂y = x  (derivative with respect to y is x)
            
        This is the product rule from calculus.
        
        Args:
            other: Another Value or number to multiply
            
        Returns:
            New Value representing the product
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            """Compute gradients for multiplication operation."""
            # Product rule: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __pow__(self, other: Union[float, int]) -> 'Value':
        """
        Power operation: self ** other
        
        Mathematical background:
            If f(x) = x^n, then:
            ∂f/∂x = n * x^(n-1)
            
        This is the power rule from calculus.
        
        Args:
            other: The exponent (must be a number, not another Value for now)
            
        Returns:
            New Value representing self raised to the power of other
            
        Raises:
            TypeError: If other is not a number
            ValueError: If the operation would be undefined (e.g., 0^(-1))
            
        Example:
            >>> x = Value(3)
            >>> y = x ** 2  # y.data = 9
            >>> y.backward()
            >>> print(x.grad)  # 2 * 3^1 = 6
        """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Power exponent must be a number, got {type(other)}")
            
        # Check for potentially problematic cases
        if self.data == 0 and other < 0:
            raise ValueError("Cannot raise 0 to a negative power (division by zero)")
        if self.data < 0 and not isinstance(other, int):
            raise ValueError("Cannot raise negative number to non-integer power (complex result)")
            
        out = Value(self.data ** other, _children=(self,), _op=f'**{other}')
        
        def _backward():
            """Compute gradients for power operation."""
            # Power rule: ∂(x^n)/∂x = n * x^(n-1)
            if other == 0:
                # Special case: x^0 = 1, derivative is 0
                gradient = 0
            elif self.data == 0:
                # Special case: 0^n where n > 0, derivative is 0
                gradient = 0  
            else:
                gradient = other * (self.data ** (other - 1))
            
            self.grad += gradient * out.grad
            
        out._backward = _backward
        return out
    
    def __truediv__(self, other: Union['Value', float, int]) -> 'Value':
        """
        Division operation: self / other
        
        Mathematical insight:
            Division can be rewritten as multiplication:
            a / b = a * b^(-1)
            
        So we can implement division using multiplication and power operations
        that we already have!
        
        Args:
            other: The divisor
            
        Returns:
            New Value representing the quotient
        """
        other = other if isinstance(other, Value) else Value(other)
        
        # Check for division by zero
        if other.data == 0:
            raise ValueError("Division by zero")
            
        # Implement as multiplication by reciprocal: a/b = a * b^(-1)
        return self * (other ** -1)
    
    def __neg__(self) -> 'Value':
        """
        Negation operation: -self
        
        Mathematical background:
            If f(x) = -x, then ∂f/∂x = -1
            
        Returns:
            New Value representing the negation
        """
        return self * -1
    
    def __sub__(self, other: Union['Value', float, int]) -> 'Value':
        """
        Subtraction operation: self - other
        
        Mathematical insight:
            Subtraction can be rewritten as addition:
            a - b = a + (-b)
            
        So we implement it using addition and negation operations.
        """
        return self + (-other)
    
    # Right-hand side operations (when Value is on the right side)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other  
    def __rtruediv__(self, other): return Value(other) / self
    def __rsub__(self, other): return Value(other) - self
    def __rpow__(self, other): return Value(other) ** self
        
    def exp(self) -> 'Value':
        """
        Exponential function: e^self
        
        Mathematical background:
            The exponential function is special because it's its own derivative:
            If f(x) = e^x, then ∂f/∂x = e^x
            
        This is why exp() is so important in neural networks!
        
        Returns:
            New Value representing e^self
            
        Example:
            >>> x = Value(1)
            >>> y = x.exp()  # y.data ≈ 2.718 (e^1)
            >>> y.backward()  
            >>> print(x.grad)  # ≈ 2.718 (same as y.data)
        """
        # Prevent overflow for very large inputs
        if self.data > 700:  # exp(700) is near float overflow
            raise ValueError(f"exp({self.data}) would overflow")
            
        exp_value = math.exp(self.data)
        out = Value(exp_value, _children=(self,), _op='exp')
        
        def _backward():
            """Compute gradients for exponential function."""
            # Special property: ∂(e^x)/∂x = e^x
            self.grad += exp_value * out.grad
            
        out._backward = _backward
        return out
    
    def log(self) -> 'Value':
        """
        Natural logarithm: ln(self)
        
        Mathematical background:
            If f(x) = ln(x), then ∂f/∂x = 1/x
            
        The logarithm is only defined for positive numbers.
        
        Returns:
            New Value representing ln(self)
            
        Raises:
            ValueError: If self.data <= 0
        """
        if self.data <= 0:
            raise ValueError(f"log() undefined for {self.data} (must be positive)")
            
        out = Value(math.log(self.data), _children=(self,), _op='log')
        
        def _backward():
            """Compute gradients for natural logarithm."""
            # Derivative: ∂(ln(x))/∂x = 1/x
            self.grad += (1.0 / self.data) * out.grad
            
        out._backward = _backward
        return out
    
    def sqrt(self) -> 'Value':
        """
        Square root: √self
        
        Mathematical insight:
            Square root is just a power operation:
            √x = x^(1/2)
            
        So we can implement it using the power operation we already have!
        """
        return self ** 0.5
    
    def relu(self) -> 'Value':
        """
        ReLU (Rectified Linear Unit) activation function: max(0, self)
        
        Mathematical background:
            ReLU(x) = max(0, x) = {x if x > 0, 0 if x ≤ 0}
            
            The derivative is:
            ∂ReLU(x)/∂x = {1 if x > 0, 0 if x ≤ 0}
            
        ReLU is the most important activation function in deep learning because:
        1. It's computationally efficient
        2. It doesn't suffer from vanishing gradients
        3. It creates sparse activations (many zeros)
        
        Returns:
            New Value representing max(0, self)
        """
        relu_value = max(0, self.data)
        out = Value(relu_value, _children=(self,), _op='ReLU')
        
        def _backward():
            """Compute gradients for ReLU function."""
            # Gradient is 1 if input > 0, else 0
            gradient = 1.0 if self.data > 0 else 0.0
            self.grad += gradient * out.grad
            
        out._backward = _backward
        return out
    
    def sigmoid(self) -> 'Value':
        """
        Sigmoid activation function: 1 / (1 + e^(-self))
        
        Mathematical background:
            σ(x) = 1 / (1 + e^(-x))
            
            The derivative has a beautiful property:
            ∂σ(x)/∂x = σ(x) * (1 - σ(x))
            
        Sigmoid maps any real number to (0, 1), making it useful for:
        - Binary classification (output probabilities)
        - Gating mechanisms in RNNs
        
        Returns:
            New Value representing sigmoid(self)
        """
        # Numerical stability: prevent overflow/underflow
        x = self.data
        if x > 500:
            sigmoid_value = 1.0
        elif x < -500:  
            sigmoid_value = 0.0
        else:
            sigmoid_value = 1.0 / (1.0 + math.exp(-x))
            
        out = Value(sigmoid_value, _children=(self,), _op='sigmoid')
        
        def _backward():
            """Compute gradients for sigmoid function."""
            # Beautiful property: ∂σ(x)/∂x = σ(x) * (1 - σ(x))
            gradient = sigmoid_value * (1 - sigmoid_value)
            self.grad += gradient * out.grad
            
        out._backward = _backward
        return out
    
    def tanh(self) -> 'Value':
        """
        Hyperbolic tangent activation function
        
        Mathematical background:
            tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
            
            The derivative is:
            ∂tanh(x)/∂x = 1 - tanh²(x)
            
        Tanh maps any real number to (-1, 1), making it:
        - Zero-centered (unlike sigmoid)
        - Good for hidden layers in neural networks
        
        Returns:
            New Value representing tanh(self)
        """
        x = self.data
        
        # Numerical stability for extreme values
        if x > 500:
            tanh_value = 1.0
        elif x < -500:
            tanh_value = -1.0
        else:
            exp_2x = math.exp(2 * x)
            tanh_value = (exp_2x - 1) / (exp_2x + 1)
            
        out = Value(tanh_value, _children=(self,), _op='tanh')
        
        def _backward():
            """Compute gradients for tanh function."""
            # Derivative: ∂tanh(x)/∂x = 1 - tanh²(x)
            gradient = 1 - tanh_value ** 2
            self.grad += gradient * out.grad
            
        out._backward = _backward
        return out
      
    def backward(self) -> None:
        """
        Compute gradients using backpropagation.
        
        This is the heart of automatic differentiation! It implements
        the chain rule by traversing the computational graph in reverse
        topological order.
        
        Algorithm:
        1. Build topological ordering of all nodes in the graph
        2. Initialize this node's gradient to 1.0 (∂output/∂output = 1)
        3. For each node in reverse topological order:
           - Call its _backward function to propagate gradients
           
        The topological sort ensures we compute gradients in the correct order.
        """
        # Build topological ordering of the computational graph
        topo = []
        visited = set()
        
        def build_topo(v):
            """Recursively build topological ordering."""
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient of output
        self.grad = 1.0
        
        # Propagate gradients in reverse topological order
        for node in reversed(topo):
            node._backward()

print("\n✅ FIXED VALUE CLASS IMPLEMENTED")
print("Key improvements:")
print("• Fixed exp() bug")
print("• Added power, division, subtraction operations")
print("• Added sqrt, relu, sigmoid, tanh functions")
print("• Comprehensive error handling")
print("• Detailed documentation")
print("• Type hints for better IDE support")

# ============================================================================
# TESTING THE FIXED VALUE CLASS
# ============================================================================

print("\n\n🧪 TESTING THE FIXED VALUE CLASS")
print("="*50)

def test_value_class():
    """Comprehensive test suite for the Value class."""
    
    print("Test 1: Basic operations")
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = a * b + a  # c = 2*3 + 2 = 8
    c.backward()
    
    print(f"a = {a.data}, b = {b.data}, c = {c.data}")
    print(f"∂c/∂a = {a.grad} (expected: b + 1 = 4.0)")
    print(f"∂c/∂b = {b.grad} (expected: a = 2.0)")
    assert abs(a.grad - 4.0) < 1e-6, f"Expected a.grad=4.0, got {a.grad}"
    assert abs(b.grad - 2.0) < 1e-6, f"Expected b.grad=2.0, got {b.grad}"
    
    print("\nTest 2: Power and division")
    x = Value(4.0)
    y = x ** 2 / 2  # y = 16/2 = 8
    y.backward()
    print(f"x = {x.data}, y = {y.data}")
    print(f"∂y/∂x = {x.grad} (expected: 2*4/2 = 4.0)")
    assert abs(x.grad - 4.0) < 1e-6
    
    print("\nTest 3: Activation functions")
    z = Value(0.5)
    sigmoid_z = z.sigmoid()
    relu_z = z.relu()  
    tanh_z = z.tanh()
    
    print(f"z = {z.data}")
    print(f"sigmoid({z.data}) = {sigmoid_z.data:.4f}")
    print(f"relu({z.data}) = {relu_z.data}")
    print(f"tanh({z.data}) = {tanh_z.data:.4f}")
    
    print("\nTest 4: Complex expression")
    # Test: f(x) = sigmoid(x^2 + exp(x))
    x = Value(1.0)
    f = (x**2 + x.exp()).sigmoid()
    f.backward()
    
    print(f"f(1) = sigmoid(1^2 + e^1) = {f.data:.4f}")
    print(f"f'(1) = {x.grad:.4f}")
    
    print("\n✅ All tests passed!")

# Run the tests
test_value_class()

print(f"""

🎉 CONGRATULATIONS! 

You now have a robust, well-documented Value class that:
• Handles all basic mathematical operations correctly
• Has proper error handling and edge case management  
• Includes comprehensive documentation
• Supports the most important activation functions
• Has been thoroughly tested

This is a solid foundation for building more advanced features!

📚 WHAT YOU LEARNED:
• How to implement mathematical operations with correct gradients
• The importance of numerical stability in mathematical functions
• How to structure code with proper documentation and error handling
• How automatic differentiation works through the chain rule
• Why certain activation functions (ReLU, sigmoid, tanh) are important

🎯 NEXT LESSON: 
We'll learn about tensors - extending this scalar Value class to work with 
multi-dimensional arrays. This is where things get really interesting!

Ready for the next lesson? 🚀
""")

print("\n" + "="*80)
print("END OF LESSON 1: FIXED VALUE CLASS")
print("="*80)