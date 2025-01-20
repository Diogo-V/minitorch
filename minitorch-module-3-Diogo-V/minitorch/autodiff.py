from dataclasses import dataclass
from typing import Any, Deque, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    x_forward = list(vals)
    x_backwards = list(vals)

    x_forward[arg] += epsilon
    x_backwards[arg] -= epsilon

    return (f(*x_forward) - f(*x_backwards)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    topo = []
    visited = set()

    def visit(v: Variable):
        if v.is_constant() or v in visited:
            return
        if not v.is_leaf():
            for parent in v.parents:
                visit(parent)
        visited.add(v)
        topo.append(v)

    visit(variable)
    return topo[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = topological_sort(variable)
    var_to_grad = {variable.unique_id: deriv}

    for var in order:
        # This means that this is not a tracked variable
        if var.unique_id not in var_to_grad:
            continue

        current_derivative = var_to_grad[var.unique_id]
        new_derivative = var.chain_rule(current_derivative)

        for child, grad in new_derivative:
            if child.is_leaf():
                child.accumulate_derivative(grad)
            else:
                if child.unique_id in var_to_grad:
                    var_to_grad[child.unique_id] += grad
                else:
                    var_to_grad[child.unique_id] = grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
