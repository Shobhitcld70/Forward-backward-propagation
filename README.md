this project implements a minimal automatic differentiation engine from scratch, similar in spirit to frameworks like PyTorch, but extremely small and easy to understand.
It supports:

Scalar values with gradients

Forward operations (+, -, *, /, **)

Activation functions (ReLU, tanh)

Full backward propagation using a topological graph

Clean and minimal API

This repository is excellent for learning how deep learning frameworks compute gradients under the hood.

🚀 Features
1. Value Class (Core Data Structure)

Every number in the computation graph is stored as a Value object. It contains:

data: the actual scalar value

grad: its gradient (computed during .backward())

_prev: references to previous nodes

_op: the operation that created it

This allows the engine to build a full computation graph during forward pass.

2. Supported Operations

The following operations are overloaded:

Operation	Method
Addition	__add__
Subtraction	__sub__
Multiplication	__mul__
Division	__truediv__
Power	__pow__
Negation	__neg__

Each operation constructs a new Value node and defines its own backward step.

3. Activation Functions

✔ ReLU — returns max(0, x)
✔ tanh — smooth squashing function

Both include their own gradient logic.

4. Backpropagation

Calling .backward() on any final node:

Performs a topological sort on the graph

Traverses nodes in reverse topological order

Applies chain rule to accumulate gradients into v.grad

This is exactly how real deep learning frameworks compute gradients.
