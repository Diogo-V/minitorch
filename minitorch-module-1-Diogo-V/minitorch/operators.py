"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiplies together two numbers.

    Args:
        a: floating number
        b: floating number

    Returns:
        Dot product of two numbers.

    """
    return a * b


def id(input: float) -> float:
    """Returns the input unchanged"""
    return input


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negates a number."""
    return float(-a)


def lt(a: float, b: float) -> float:
    """Checks if one number is less than another."""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal.

    Args:
        a: floating point number
        b: floating point number

    Returns:
        True if they are equal and False otherwise.

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the lasrgest of the two numbers.

    Args:
        a: floating point number
        b: floating point number

    Returns:
        Largest between both numbers

    """
    return b if lt(a, b) else a


def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value.

    Args:
        a: floating point number
        b: floating point number

    Returns:
        True if they are close and False otherwise

    """
    return lt(abs(a - b), 1e-2)


def sigmoid(a: float) -> float:
    """Calculates the sigmoid value of the input number.

    Args:
        a: floating point number

    Returns:
        Sigmoid of the input

    """
    if lt(a, 0):
        return (exp(a)) / (1 + exp(a))
    else:
        return (1) / (1 + exp(neg(a)))


def relu(a: float) -> float:
    """Applies ReLu operation on the number."""
    return max(a, 0.0)


def log(a: float) -> float:
    """Computes the logarith of the input."""
    return math.log(a)


def exp(a: float) -> float:
    """Calculates the exponential of the input."""
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal (inverse) of a value.

    Args:
        a: Value to obtain the inverse from

    Returns:
        Inverse value.

    """
    return 1 / a


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
        a: Input to log function
        b: Multiplier value to the derivative of log

    Returns:
        Computed derivative.

    """
    return b / a


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
        a: Value that was inverse
        b: Multiplier value to the derivative of the reciprocal

    Returns:
        Computed derivative.

    """
    return b * (-1.0 / a**2)


def relu_back(a: float, b: float) -> float:
    """Computes the derivative of relu times a second arg.

    Args:
        a: ReLu function argument
        b: Second argument to be Multiplier

    Returns:
        Computed derivative.

    """
    return b * (0.0 if lt(a, 0) else 1.0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable, l: Iterable) -> Iterable:
    """Applies a given function to each element of an itereable.

    Args:
        fn: Function to be applied
        l: List to which to apply the function

    Returns:
        New list after applying function to all elements.

    """
    new_l = []
    for e in l:
        new_l.append(fn(e))
    return new_l


def zipWith(fn: Callable, l1: Iterable, l2: Iterable) -> Iterable:
    """Combines two iterables using a given function.

    Args:
        fn: Function to be applied
        l1: List to be combined
        l2: Anpother list to be combined

    Returns:
        New combined list

    """
    new_l = []

    for l1_e, l2_e in zip(l1, l2):
        new_l.append(fn(l1_e, l2_e))

    return new_l


def reduce(fn: Callable, l: Iterable[float], init: float) -> float:
    """Reduces a single list into a value using a function.

    Args:
        fn: Function to be applied
        l: List to be combined
        init: initial accumulator value

    Returns:
        Computed value

    """
    for e in l:
        init = fn(e, init)
    return init


def negList(l: Iterable) -> Iterable:
    """Negates all values in a list."""
    return map(lambda x: -x, l)


def addLists(l1: Iterable, l2: Iterable) -> Iterable:
    """Adds values in two lists to be a single list."""
    return zipWith(lambda x, y: x + y, l1, l2)


def sum(l: Iterable) -> float:
    """Sums all values in a list."""
    return reduce(lambda x, y: x + y, l, 0)


def prod(l: Iterable) -> float:
    """Multiplies all values in a list."""
    return reduce(lambda x, y: x * y, l, 1)
