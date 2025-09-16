from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol


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
    val_plus_eps = list(vals)
    val_minus_eps = list(vals)

    val_plus_eps[arg] = val_plus_eps[arg] + epsilon
    val_minus_eps[arg] = val_minus_eps[arg] - epsilon

    central_diff = (f(*val_plus_eps) - f(*val_minus_eps)) / (2 * epsilon)
    return central_diff


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
    visited_ids = []

    def dfs(variable: Variable, ordered: list[Variable]):
        visited_ids.append(variable.unique_id)

        for dep in variable.history.inputs:
            if dep.unique_id not in visited_ids:
                dfs(dep, ordered)

        ordered.append(variable)
        return

    ordered = []
    dfs(variable, ordered)
    return ordered


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    scalars = {variable.unique_id: deriv}
    ordered = topological_sort(variable)

    for node in reversed(ordered):
        if not node.is_leaf():
            derivatives = node.chain_rule(scalars[node.unique_id])

            for variable, deriv in derivatives:
                if variable.unique_id not in scalars:
                    scalars[variable.unique_id] = 0
                scalars[variable.unique_id] += deriv

        if node.is_leaf():
            node.accumulate_derivative(scalars[node.unique_id])

    return


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
