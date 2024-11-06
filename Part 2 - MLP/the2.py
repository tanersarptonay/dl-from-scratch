import random
import math
from typing import Union, Optional, Tuple, List
import matplotlib.pyplot as plt
import datetime


# region 1 The Gergen Library

# region random functions
def cekirdek(sayi: int):
    random.seed(sayi)


def rastgele_dogal(boyut, aralik=None, dagilim='uniform'):
    """
    Generates data of specified dimensions with random integer values and returns a gergen object.

    Parameters:
    boyut (tuple): Shape of the desired data.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to None, which implies a default range.
    dagilim (string, optional): Distribution of random values ('uniform' or other types). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random integer values.
    """

    # Set a default range if aralik is not provided
    if aralik is None:
        aralik = (0, 10)

    def generate_random_data(shape):
        if len(shape) == 1:
            return [random_value(aralik, dagilim) for _ in range(shape[0])]
        else:
            return [generate_random_data(shape[1:]) for _ in range(shape[0])]

    def random_value(aralik, dagilim):
        if dagilim == 'uniform':
            return random.randint(*aralik)
        else:
            raise ValueError(f"Unsupported distribution: {dagilim}")

    data = generate_random_data(boyut)
    return gergen(data)


def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    """
    Generates a gergen of specified dimensions with random floating-point values.

    Parameters:
    boyut (tuple): Shape of the desired gergen.
    aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0.0, 1.0) for uniform distribution.
    dagilim (string, optional): Distribution of random value (e.g., 'uniform', 'gaussian'). Defaults to 'uniform'.

    Returns:
    gergen: A new gergen object with random floating-point values.
    """

    def generate_random_data(shape):
        if len(shape) == 1:
            return [random_value(aralik, dagilim) for _ in range(shape[0])]
        else:
            return [generate_random_data(shape[1:]) for _ in range(shape[0])]

    def random_value(aralik, dagilim):
        if dagilim == 'uniform':
            return random.uniform(*aralik)
        elif dagilim == 'gaussian':
            mean, std_dev = aralik
            return random.gauss(mean, std_dev)
        else:
            raise ValueError(f"Unsupported distribution: {dagilim}")

    data = generate_random_data(boyut)
    return gergen(data)


# endregion random functions

# region Gergen Operations
class Operation:
    def __init__(self):
        self.called = False

    def __call__(self, *operands, **kwargs):
        """
        Modified to accept keyword arguments as well.
        """
        self.operands = operands
        self.kwargs = kwargs  # Store keyword arguments separately
        self.outputs = self.ileri(*operands, **kwargs)
        self.called = True
        return self.outputs

    def __str__(self):
        info_str = self.__class__.__name__ + " Operation with\n"

        if not self.called:
            return info_str + "No operands provided."

        for operand in self.operands:
            info_str += f"{operand} \n\n"
        info_str = info_str[:-2]

        for kwarg in self.kwargs:
            info_str += f"{kwarg}: {self.kwargs[kwarg]}\n"

        return info_str

    def ileri(self, *operands, **kwargs):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.
            **kwargs: Variable length keyword argument list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError

    def geri(self, grad_input):
        """
        Defines the backward pass of the operation.
        Must be implemented by subclasses to compute the gradients.

        Parameters:
            grad_input: The gradient of the loss w.r.t. the output of this operation.

        """
        raise NotImplementedError


class Add(Operation):
    def ileri(self, a, b) -> 'gergen':
        """
        Adds two gergen objects or a gergen object and a scalar.
        You can modify this function.
        """
        self.a = a
        self.b = b
        self.operands = [a, b]

        if isinstance(a, gergen) and isinstance(b, gergen):
            result = gergen(self.add_gergen(a.duzlestir().listeye(), b.duzlestir().listeye()), operation=self)
            result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (list)):
            result = gergen(self.add_list(a.listeye(), b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (list)):
            result = gergen(self.add_list(b.listeye(), a), operation=self)
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            result = gergen(self.add_scalar(a.listeye(), b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            result = gergen(self.add_scalar(b.listeye(), a), operation=self)
        else:
            raise ValueError(f"Add operation requires at least one gergen operand. {type(a)} and {type(b)} provided.")

        return result

    def add_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.add_scalar(elem, scalar) for elem in a]
        else:
            return a + scalar

    def add_gergen(self, a, b):
        # Check if 'a' is a list
        if isinstance(a, list):
            # Check if 'b' is a list
            if isinstance(b, list):
                if len(a) != len(b):
                    raise ValueError(f"Dimensions of gergen objects do not match for addition. {len(a)} != {len(b)}")
                return [a[i] + b[i] for i in range(len(a))]
            # If 'a' is a list and 'b' is a scalar
            elif not isinstance(b, list):
                return [item + b for item in a]

        # If 'a' is a scalar and 'b' is a list
        elif not isinstance(a, list) and isinstance(b, list):
            return [a + item for item in b]
        # Direct addition for scalars, or fallback error for unsupported types
        elif not isinstance(a, list) and not isinstance(b, list):
            return a + b

    def add_list(self, a, b):
        # Check if 'a' is a list
        if isinstance(a, list) and isinstance(b, list):
            return [self.add_list(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # If 'a' is list and b is scalar
        elif isinstance(a, list) and not isinstance(b, list):
            return [self.add_list(elem_a, b) for elem_a in a]
        elif not isinstance(a, list) and isinstance(b, list):
            return [self.add_list(a, elem_b) for elem_b in b]
        elif not isinstance(a, list) and not isinstance(b, list):
            return a + b

    def geri(self, grad_input):
        """
        Backward pass for the Add operation.
        Args:
            grad_input: The gradient of the loss w.r.t. the output of this operation.
        Returns:
            The gradient of the loss w.r.t. each of the operands.
        """
        '''
        In the Add operation, we have two
        operands, let us say a and b. The ileri() function returns the result as r = a+b. In the
        geri() function, we need to calculate:
        grad_input * (derivative of r wrt to a) and grad_input * (derivative of r wrt to b).
        grad_input * (derivative of r wrt to a) is the gradient input that should be passed to gergen a and
        grad_input * (derivative of r wrt to b) is to gergen b.
        '''

        self.operands[0].turev = grad_input

        self.operands[0].turev.requires_grad = False
        if self.operands[0].operation is not None:
            self.operands[0].operation.geri(self.operands[0].turev)

        self.operands[1].turev = grad_input

        self.operands[1].turev.requires_grad = False
        if self.operands[1].operation is not None:
            self.operands[1].operation.geri(self.operands[1].turev)


class Sub(Operation):
    """
    Subtracts two gergen objects or a gergen object and a scalar.
    You can modify this function.
    """

    def ileri(self, a, b) -> 'gergen':
        if isinstance(a, gergen) and isinstance(b, gergen):
            self.a, self.b = a, b
            self.operands = [a, b]
            result = gergen(self.subtract_gergen(a.duzlestir().listeye(), b.duzlestir().listeye()), operation=self)
            result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (list)):
            self.a = a
            self.operands = [a]
            result = gergen(self.subtract_list(a.listeye(), b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (list)):
            self.b = b
            self.operands = [b]
            result = gergen(self.subtract_list(a, b.listeye()), operation=self)
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            self.b = b
            self.operands = [a]
            result = gergen(self.subtract_scalar(a.listeye(), b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            self.b = b
            self.operands = [b]
            result = gergen(self.subtract_scalar(b.listeye(), a), operation=self)
        else:
            raise ValueError(f"Sub operation requires at least one gergen operand. {type(a)} and {type(b)} provided.")
        return result

    def subtract_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.subtract_scalar(elem, scalar) for elem in a]
        else:
            return a - scalar

    def subtract_list(self, a, b):
        # Check if 'a' is a list
        if isinstance(a, list) and isinstance(b, list):
            return [self.subtract_list(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # If 'a' is list and b is scalar
        elif isinstance(a, list) and not isinstance(b, list):
            return [self.subtract_list(elem_a, b) for elem_a in a]
        elif not isinstance(a, list) and isinstance(b, list):
            return [self.subtract_list(a, elem_b) for elem_b in b]
        elif not isinstance(a, list) and not isinstance(b, list):
            return a - b

    def subtract_gergen(self, a, b):
        # Check if 'a' is a list
        if isinstance(a, list):
            # Check if 'b' is a list
            if isinstance(b, list):
                if len(a) != len(b):
                    raise ValueError(f"Dimensions of gergen objects do not match for subtraction. {len(a)} != {len(b)}")
                return [a[i] - b[i] for i in range(len(a))]
            # If 'a' is a list and 'b' is a scalar
            elif not isinstance(b, list):
                return [item - b for item in a]

        # If 'a' is a scalar and 'b' is a list
        elif not isinstance(a, list) and isinstance(b, list):
            return [a - item for item in b]
        # Direct subtraction for scalars, or fallback error for unsupported types
        elif not isinstance(a, list) and not isinstance(b, list):
            return a - b

    def geri(self, grad_input):
        '''
        TODO: Implement the gradient computation for the Sub operation.
        '''
        pass


class TrueDiv(Operation):
    """
    Divides two gergen objects or a gergen object and a scalar.
    You can modify this function.
    """

    def ileri(self, a, b) -> 'gergen':
        if isinstance(a, gergen) and isinstance(b, gergen):
            self.a, self.b = a, b
            self.operands = [a, b]
            result = gergen(self.divide_elements(a.duzlestir().listeye(), b.duzlestir().listeye()), operation=self)
            result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            self.a = a
            self.operands = [a]
            result = gergen(self.divide_scalar(a.listeye(), b), operation=self)

        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            # Division of a scalar by a gergen object is not typically defined,
            # but you can implement it based on your requirements.
            raise NotImplementedError("Division of a scalar by a gergen object is not implemented.")
        else:
            raise ValueError(
                f"TrueDiv operation requires at least one gergen operand. {type(a)} and {type(b)} provided.")

        return result

    def divide_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.divide_scalar(elem, scalar) for elem in a]
        else:
            if scalar == 0:
                raise ZeroDivisionError("Division by zero.")
            return a / scalar

    def divide_elements(self, a, b):
        # Both a and b are non-lists (scalars), perform direct division
        if not isinstance(a, list) and not isinstance(b, list):
            if b == 0:
                raise ZeroDivisionError("Division by zero.")
            return a / b
        # Both a and b are lists, perform element-wise division
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError(f"Dimensions of gergen objects do not match for division. {len(a)} != {len(b)}")
            return [self.divide_elements(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # One of a or b is a list and the other is a scalar, divide each element of the list by the scalar
        elif isinstance(a, list):
            return [self.divide_elements(elem, b) for elem in a]
        else:
            raise NotImplementedError(f"Division of scalar by a list is not typically defined. {a} / {b}")

    def geri(self, grad_input):
        '''
        TODO (Optional): Implement the gradient computation for the TrueDiv operation.
        '''
        pass


class Mul(Operation):
    """
    Multiplies two gergen objects or a gergen object and a scalar.
    You can modify this function.
    """

    def ileri(self, a, b) -> 'gergen':
        if isinstance(a, gergen) and isinstance(b, gergen):
            self.a, self.b = a, b
            self.operands = [a, b]
            # a is a scalar gergen
            if a.uzunluk() == 1:
                result = gergen(self.multiply_scalar(b.listeye(), a.listeye()), operation=self)
            # b is a scalar gergen
            elif b.uzunluk() == 1:
                result = gergen(self.multiply_scalar(a.listeye(), b.listeye()), operation=self)
            else:
                result = gergen(self.multiply_elements(a.duzlestir().listeye(), b.duzlestir().listeye()),
                                operation=self)
                result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            self.a = a
            self.b = b
            self.operands = [a]
            result = gergen(self.multiply_scalar(a.listeye(), b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            self.b = b
            self.b = a
            self.operands = [b]
            result = gergen(self.multiply_scalar(b.listeye(), a), operation=self)
        else:
            raise ValueError(f"Mul operation requires at least one gergen operand. {type(a)} and {type(b)} provided.")

        return result

    def multiply_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.multiply_scalar(elem, scalar) for elem in a]
        else:
            return a * scalar

    def multiply_elements(self, a, b):
        # Both a and b are non-lists (scalars), perform direct multiplication
        if not isinstance(a, list) and not isinstance(b, list):
            return a * b
        # Both a and b are lists, perform element-wise multiplication
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError(f"Dimensions of gergen objects do not match for multiplication. {len(a)} != {len(b)}")
            return [self.multiply_elements(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # One of a or b is a list and the other is a scalar, multiply each element of the list by the scalar
        elif isinstance(a, list):
            return [self.multiply_elements(elem, b) for elem in a]
        else:
            return [self.multiply_elements(a, elem) for elem in b]

    def geri(self, grad_input):
        '''
        Backward pass for the Multiplication operation.
        Args:
            grad_input: The gradient of the loss w.r.t. the output of this operation.
        Returns:
            The gradient of the loss w.r.t. each of the operands.
        '''

        if self.operands[0].requires_grad:
            a_grad = grad_input * self.operands[1]
        else:
            a_grad = None

        if self.operands[1].requires_grad:
            b_grad = grad_input * self.operands[0]
        else:
            b_grad = None

        self.operands[0].turev = a_grad
        self.operands[1].turev = b_grad

        return a_grad, b_grad


class Us(Operation):
    """
    Power operation.
    You can modify this function.
    """

    def ileri(self, a, n) -> 'gergen':
        self.a = a
        self.n = n
        self.operands = [a]
        result = gergen(self.power_elements(a.listeye(), n), operation=self)
        return result

    def power_elements(self, a, n):

        if isinstance(a, list):
            return [self.power_elements(elem, n) for elem in a]
        else:
            return a ** n

    def multiply_elements(self, a, b):
        # Both a and b are non-lists (scalars), perform direct multiplication
        if not isinstance(a, list) and not isinstance(b, list):
            return a * b
        # Both a and b are lists, perform element-wise multiplication
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError(f"Dimensions of gergen objects do not match for multiplication. {len(a)} != {len(b)}")
            return [self.multiply_elements(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # One of a or b is a list and the other is a scalar, multiply each element of the list by the scalar
        elif isinstance(a, list):
            return [self.multiply_elements(elem, b) for elem in a]
        else:
            return [self.multiply_elements(a, elem) for elem in b]

    def geri(self, grad_input):
        '''
        TODO: Implement the gradient computation for the Power operation.
        '''
        pass


class Log10(Operation):
    """
    Log10 operation
    You can modify this function.
    """

    def ileri(self, a) -> 'gergen':
        self.a = a
        self.operands = [a]
        # Recursively check for non-positive values in the nested list structure
        if self.contains_non_positive(self.a.listeye()):
            raise ValueError("Logarithm undefined for non-positive values.")
        result = gergen(self.log_elements(a.listeye()), operation=self)
        return result

    def log_elements(self, a):
        # Recursively apply the base 10 logarithm to each element
        if isinstance(a, list):
            return [self.log_elements(elem) for elem in a]
        else:
            try:
                return math.log10(a)
            except ValueError:
                return math.log10(0)

    def contains_non_positive(self, a):
        # Recursively check for non-positive values and flatten the results
        def check_and_flatten(a):
            flag = False
            if isinstance(a, list):
                # Use a generator expression to recursively check each element and flatten the result
                for ele in a:
                    flag = check_and_flatten(ele)
            else:
                if a <= 0:
                    return True
            return flag

        # Use 'any' on a flattened generator of boolean values
        return check_and_flatten(a)

    def multiply_elements(self, a, scalar):
        # Recursively multiply each element by the scalar
        if isinstance(a, list):
            return [self.multiply_elements(elem, scalar) for elem in a]
        else:
            return a * scalar

    def divide_elements(self, grad_output, b):
        # Recursively divide grad_output by b, assuming they have the same structure
        if isinstance(b, list):
            return [self.divide_elements(elem_grad, elem_b) for elem_grad, elem_b in zip(grad_output, b)]
        else:
            return grad_output / b

    def geri(self, grad_input):
        '''
        TODO (Optional): Implement the gradient computation for the Log10 operation.
        '''
        pass


class Ln(Operation):
    def ileri(self, a) -> 'gergen':
        """
        Implements the forward pass for the Ln operation.
        You can modify this function.
        """
        if not isinstance(a, gergen):
            raise ValueError(f"Ln operation requires a gergen operand. {type(a)} provided.")
        self.a = a
        self.operands = [a]
        if self.contains_non_positive(self.a.listeye()):
            raise ValueError("Logarithm undefined for non-positive values.")

        result = gergen(self.log_elements(a.listeye()), operation=self)
        return result

    def log_elements(self, a):
        # Recursively apply the base 10 logarithm to each element
        if isinstance(a, list):
            return [self.log_elements(elem) for elem in a]
        else:
            return math.log(a) if a > 0 else math.log(a + 10 ** -4)

    def contains_non_positive(self, a):
        # Recursively check for non-positive values
        def check_and_flatten(a):
            if isinstance(a, list):
                return any(check_and_flatten(elem) for elem in a)
            else:
                if a <= 0:
                    a = 1
                    return True
                else:
                    return False

        # Use 'any' on a flattened generator of boolean values
        return check_and_flatten(a)

    def geri(self, grad_input):
        '''
        TODO: Implement the gradient computation for the Ln operation.
        '''
        pass


def apply_elementwise(g, func):
    """
    Applies a given function element-wise to the data in a gergen object.
    This version is capable of handling nested lists of any depth.
    """

    def recursive_apply(data):
        if isinstance(data, list):
            # Recursively apply func to each element if data is a list
            return [recursive_apply(sublist) for sublist in data]
        else:
            # Apply func directly if data is a scalar (non-list)
            return func(data)

    # Use the recursive function to apply the operation to the gergen object's data
    return recursive_apply(g.listeye())


class Sin(Operation):
    def ileri(self, a) -> 'gergen':
        """
        Implements the forward pass for the Sin operation.
        You can modify this function.
        """
        self.operands = [a]
        result = gergen(apply_elementwise(a, math.sin), operation=self)
        return result

    def geri(self, grad_output):
        """
        TODO(Optional): Implement the gradient computation for the Sin operation.
        """
        pass


class Cos(Operation):
    def ileri(self, a) -> 'gergen':
        """
        Implements the forward pass for the Cos operation.
        You can modify this function.
        """
        self.operands = [a]
        result = gergen(apply_elementwise(a, math.cos), operation=self)
        return result

    def geri(self, grad_output):
        """
        TODO(Optional): Implement the gradient computation for the Cos operation.
        """
        pass


class Tan(Operation):
    def ileri(self, a) -> 'gergen':
        """
        Implements the forward pass for the Tan operation.
        You can modify this function.
        """
        self.operands = [a]
        result = gergen(apply_elementwise(a, math.tan), operation=self)
        return result

    def geri(self, grad_output):
        """
        TODO(Optional): Implement the gradient computation for the Tan operation.
        """
        pass


class Topla(Operation):
    def ileri(self, a, eksen=None) -> 'gergen':
        """
        Forward pass for the Topla operation.
        You can modify this function.
        """

        def sum_elements(lst):
            if isinstance(lst[0], list):
                return [sum_elements(sublst) for sublst in zip(*lst)]
            else:
                return sum(lst)

        def sum_along_axis(data, axis):
            if axis == 0:
                return sum_elements(data)
            else:
                return [sum_along_axis(subdata, axis - 1) for subdata in data]

        self.operands = [a]
        if eksen is None:
            result = sum(a.duzlestir().listeye())
        elif isinstance(eksen, int):
            if eksen < 0 or eksen >= len(a.boyut()):
                raise ValueError(f"Axis out of bounds for gergen's dimensionality ({eksen} not in {a.boyut()})")
            result = sum_along_axis(a.listeye(), eksen)
        else:
            raise TypeError(f"Axis must be an integer or None ({type(eksen)} provided)")

        return gergen(result, operation=self)

    def geri(self, grad_input):
        """
        TODO(Optional): Implement the gradient computation for the Topla operation.
        """
        pass


class Ortalama(Operation):
    def ileri(self, a, eksen=None) -> Union['gergen', float, int]:
        """
        Forward pass for the Ortalama operation.
        """

        def average_elements(total_sum, total_elements):
            # Compute the average
            if isinstance(total_sum, list):
                # If total_sum is a list (multi-dimensional case), calculate the average for each sublist
                return [average_elements(ts, total_elements) for ts in total_sum]
            else:
                # For a single number, just divide
                return total_sum / total_elements

        self.operands = [a]
        sum_op = Topla()  # Instantiate the Sum operation

        total_sum = sum_op.ileri(a, eksen=eksen).listeye()

        if eksen is None:
            total_elements = a.uzunluk()
        else:
            if eksen < 0 or eksen >= len(a.boyut()):
                raise ValueError(f"Axis out of bounds for gergen's dimensionality ({eksen} not in {a.boyut()})")
            total_elements = a.boyut()[eksen]

        # Compute the average
        average_result = average_elements(total_sum, total_elements)

        return gergen(average_result, operation=self)

    def geri(self, grad_input):
        """
        TODO: Implement the gradient computation for the Ortalama operation.
        """
        pass


class IcCarpim(Operation):
    def ileri(self, a, b) -> 'gergen':
        """
        Forward pass for the inner product operation.
        """
        self.a = a
        self.b = b
        self.operands = [a, b]
        if not isinstance(a, type(b)):
            raise ValueError(f"Both operands must be gergen objects. {type(a)} and {type(b)} provided.")

        def is_vector(v):
            return len(v.boyut()) == 1

        def is_matrix(m):
            return len(m.boyut()) == 2

        def vector_dot_product(v1, v2):
            if len(v1) != len(v2):
                raise ValueError(f"Vectors must have the same length for dot product. {len(v1)} != {len(v2)}")
            return sum(x * y for x, y in zip(v1, v2))

        def matrix_multiply(m1, m2):
            if len(m1[0]) != len(m2):
                raise ValueError(
                    f"The number of columns in the first matrix must match the number of rows in the second matrix for iç çarpım. Gergens with {a.boyut()} and {b.boyut()} provided.")
            return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*m2)] for row_a in m1]

        if len(a.boyut()) > 2 or len(b.boyut()) > 2:
            raise ValueError(
                f"Operands must both be either 1-D vectors or 2-D matrices. {a.boyut()} and {b.boyut()} provided.")
        elif is_vector(a) and is_vector(b):
            # Perform vector dot product
            result = vector_dot_product(a.listeye(), b.listeye())
        elif is_matrix(a) and is_matrix(b):
            # Perform matrix multiplication
            result = matrix_multiply(a.listeye(), b.listeye())
        else:
            raise ValueError(
                f"Operands must both be either 1-D vectors or 2-D matrices. {a.boyut()} and {b.boyut()} provided.")

        # Return result
        return gergen(result, operation=self)

    def geri(self, grad_input):
        """
        Backward pass for the operation explained above.
        dr/da = b^T
        dr/db = a^T

        with grad_input:
            first: grad_input @ b^T
            second: a^T @ grad_input
        """
        a_grad = self.b.devrik()
        b_grad = self.a.devrik()

        self.operands[0].turev = grad_input @ a_grad
        self.operands[0].turev.requires_grad = False
        if self.operands[0].operation is not None:
            self.operands[0].operation.geri(self.operands[0].turev)

        self.operands[1].turev = b_grad @ grad_input
        self.operands[1].turev.requires_grad = False
        if self.operands[1].operation is not None:
            self.operands[1].operation.geri(self.operands[1].turev)
        pass


class MatMul(Operation):
    def ileri(self, a, b) -> 'gergen':
        """
        Computes the matrix product of two gergen objects.
        For example: If a is a 2x3 matrix and b is a 4x3 matrix,
        then this will calculate the inner product of a and b transpose (a * b^T).
        Args:
            a (gergen): The first matrix operand.
            b (gergen): The second matrix operand.
        Returns:
            gergen: A new gergen object containing the matrix product of a and b.
        """
        print("\n\nUSE IC CARPIM INSTEAD\n\”")
        self.a = a
        self.b = b
        self.operands = [a, b]

        ic_carpim_op = IcCarpim()

        # Perform the matrix multiplication
        result = ic_carpim_op.ileri(a, b.devrik()).listeye()

        return gergen(result, operation=self)

    def geri(self, grad_input):
        """
        Backward pass for the operation explained above.
        dr/da = b^T
        dr/db = a^T

        with grad_input:
            first: grad_input @ b^T
            second: a^T @ grad_input
        """
        print("\n\nUSE IC CARPIM INSTEAD\n\n”")
        a_grad = self.b.devrik()
        b_grad = self.a.devrik()

        self.operands[0].turev = grad_input @ a_grad
        self.operands[0].turev.requires_grad = False
        if self.operands[0].operation is not None:
            self.operands[0].operation.geri(self.operands[0].turev)

        self.operands[1].turev = b_grad @ grad_input
        self.operands[1].turev.requires_grad = False
        if self.operands[1].operation is not None:
            self.operands[1].operation.geri(self.operands[1].turev)


class DisCarpim(Operation):
    def ileri(self, a, b) -> 'gergen':
        """
        Computes the outer product of two gergen objects.
        """

        if not isinstance(a, gergen) or not isinstance(b, gergen):
            raise ValueError(f"Both operands must be gergen objects. {type(a)} and {type(b)} provided.")

        # Ensure the veri attributes are lists representing vectors
        if not all(isinstance(x, (int, float)) for x in a.listeye()) or not all(
                isinstance(y, (int, float)) for y in b.listeye()):
            raise ValueError(
                f"Both gergen objects must contain 1-D numerical data. {a.listeye()} and {b.listeye()} provided.")

        self.operands = [a, b]
        # Compute the outer product
        result = [[x * y for y in b.listeye()] for x in a.listeye()]

        # Return a new gergen object with the outer product as its veri
        return gergen(result, operation=self)

    def geri(self, grad_input):
        """
        TODO(Optional): Implement the gradient computation for the Dis_Carpim operation.
        """
        pass


# endregion Gergen Operations

# region Gergen Class
class gergen:
    # TODO: You should modify this class implementation

    __veri = None  # A nested list of numbers representing the data
    D = None  # Transpose of data
    turev = None  # Stores the derivate
    operation = None  # Stores the operation that produced the gergen
    __boyut = None  # Dimensions of the gergen (Shape)
    requires_grad = True  # Flag to determine if the gradient should be computed

    def __init__(self, veri=None, operation=None, requires_grad=None):
        # The constructor for the 'gergen' class.
        if veri is None:
            self.__veri = []
            self.__boyut = (0,)
            self.D = None
        else:
            self.__veri = veri
            self.__boyut = self.get_shape(veri, ())  # Assuming rectangular data
            self.operation = operation
            self.requires_grad = requires_grad if requires_grad is not None else self.requires_grad
            self.D = None

    def __iter__(self):
        # The __iter__ method returns the iterator object itself.
        # You can reset the iterator here if you want to allow multiple passes over the data.
        self.__iter_index = 0
        return self

    def __next__(self):
        # The __next__ method should return the next value from the iterator.
        # When there are no more elements, raise the StopIteration exception.
        if self.__iter_index < len(self.__veri):
            result = self.__veri[self.__iter_index]
            self.__iter_index += 1
            return gergen(result)
        else:
            raise StopIteration

    def __getitem__(self, key):
        """
        Allows for indexing or slicing the gergen object's data.

        Parameters:
        key (int, slice, tuple): An integer or slice for one-dimensional indexing,
                                    or a tuple for multi-dimensional indexing/slicing.

        Returns:
        The element or a new gergen object corresponding to the provided key.
        """

        # Helper function to handle recursive indexing/slicing
        def index_or_slice(data, key):
            if isinstance(key, int) or isinstance(key, slice):
                return data[key]
            elif isinstance(key, tuple):
                result = data
                for k in key:
                    result = index_or_slice(result, k)
                return result
            else:
                raise TypeError(f"Invalid index type: {type(key)}")

        # Perform the indexing or slicing operation
        result = index_or_slice(self.__veri, key)

        # If the result is a list, return it wrapped in a new gergen object
        return gergen(result)

    def __str__(self):
        # Generates a string representation
        if self.uzunluk() == 0:
            return "Empty Gergen"
        else:
            shape_str = ""
            for b in self.boyut():
                shape_str += str(b) + "x"
            if shape_str == "":
                shape_str += "0x"
            return shape_str[:-1] + " boyutlu gergen:" + "\n" + self.str_helper(self.listeye(), len(self.boyut()))

    def __len__(self):
        # Returns the length of the gergen object
        return self.uzunluk()

    def str_helper(self, data, shape, depth=0):
        if not shape:
            return str(data)
        elif not isinstance(data[0], list):
            return str(data)
        else:
            inner_results = []
            for subdata in data:
                inner_results.append(self.str_helper(subdata, shape, depth + 1))

            result = "[" + ("\n" * (shape - depth - 1)).join(r for r in inner_results) + "]"
            return result

    @staticmethod
    def get_shape(lst, shape=()):
        if not isinstance(lst, list):
            # base case
            return shape
        # peek ahead and assure all lists in the next depth
        # have the same length
        if isinstance(lst[0], list):
            l = len(lst[0])
            if not all(len(item) == l for item in lst):
                msg = 'not all lists have the same length'
                raise ValueError(msg)

        shape += (len(lst),)
        # recurse
        shape = gergen.get_shape(lst[0], shape)

        return shape

    @staticmethod
    def custom_zeros(shape):
        """
        Creates a multi-dimensional array of zeros with the specified shape.

        Parameters:
        shape (tuple): A tuple representing the dimensions of the array.

        Returns:
        A nested list (multi-dimensional array) filled with zeros.
        """
        if not shape:  # If shape is empty or reaches the end of recursion
            return 0
        # Recursively build nested lists
        return [gergen.custom_zeros(shape[1:]) for _ in range(shape[0])]

    # HELPER
    @staticmethod
    def prod(iterable):
        """Utility function to calculate the product of elements in an iterable."""
        result = 1
        for i in iterable:
            result *= i
        return result

    def __matmul__(self, other: 'gergen') -> 'gergen':
        inner_product_operation = IcCarpim()
        result_gergen = inner_product_operation(self, other)
        return result_gergen

    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        mul_operation = Mul()
        result_gergen = mul_operation(self, other)
        return result_gergen

    def __rmul__(self, other: Union['gergen', int, float]) -> 'gergen':
        mul_operation = Mul()
        result_gergen = mul_operation(self, other)
        return result_gergen

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        div_operation = TrueDiv()
        result_gergen = div_operation(self, other)
        return result_gergen

    def __rtruediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        div_operation = TrueDiv()
        result_gergen = div_operation(self, other)
        return result_gergen

    def __add__(self, other):
        add_operation = Add()
        result_gergen = add_operation(self, other)
        return result_gergen

    def __radd__(self, other):
        add_operation = Add()
        result_gergen = add_operation(self, other)
        return result_gergen

    def __sub__(self, other):
        sub_operation = Sub()
        result_gergen = sub_operation(self, other)
        return result_gergen

    def __rsub__(self, other):
        sub_operation = Sub()
        result_gergen = sub_operation(other, self)
        return result_gergen

    def copy(self):
        return gergen(self.__veri)

    def uzunluk(self):
        # Returns the total number of elements in the gergen
        total = 1
        for ele in self.__boyut:
            total *= ele
        return total

    def boyut(self):
        # Returns the shape of the gergen
        return self.__boyut

    def devrik(self):
        # Returns the transpose of gergen
        # Check if the gergen object is scalar
        if self.uzunluk() == 1:
            return gergen(self.__veri)
        # Check if the gergen object represents a 1D list (vector)
        if isinstance(self.__veri, list) and all(not isinstance(item, list) for item in self.__veri):
            # Convert each element into a list (column vector)
            return gergen([[item] for item in self.__veri])
        else:
            # Handle higher-dimensional cases (e.g., 2D matrices, 3D tensors, etc.)
            new_boyut = tuple(reversed(self.__boyut))
            order = list(reversed(range(len(self.__boyut))))
            arr = self.custom_zeros(new_boyut)  # Assuming custom_zeros initializes an array with the given shape
            paths = [0] * len(self.__boyut)
            while paths[0] < self.__boyut[0]:
                ref = self.listeye()
                place = arr
                for i in range(len(paths) - 1):
                    ref = ref[paths[i]]
                    place = place[paths[order[i]]]

                place[paths[order[-1]]] = ref[paths[-1]]
                paths[-1] += 1
                for i in range(len(paths) - 1, 0, -1):
                    if paths[i] >= self.__boyut[i]:
                        paths[i] = 0
                        paths[i - 1] += 1
                    else:
                        break
            self.D = gergen(arr)
            return gergen(arr)

    def L1(self):
        # Calculates and returns the L1 norm
        flattened_data = self.duzlestir().__veri  # Assuming flatten returns a gergen object

        # Calculate the L1 norm by summing the absolute values of elements in the flattened list
        l1_norm = sum(abs(item) for item in flattened_data)

        return l1_norm

    def L2(self):
        # Assuming flatten returns a gergen object and __veri holds the flattened data
        flattened_data = self.duzlestir().__veri

        # Calculate the L2 norm by summing the squares of elements in the flattened list and then taking the square root
        l2_norm = sum(item ** 2 for item in flattened_data) ** 0.5

        return l2_norm

    def Lp(self, p):
        # Calculates and returns the Lp norm, where p should be positive integer
        if p <= 0:
            raise ValueError("p must be a positive integer for Lp norm.")
        # Assuming flatten returns a gergen object and __veri holds the flattened data
        flattened_data = self.duzlestir().__veri

        # Calculate the Lp norm by raising elements to the power of p, summing, and then taking the p-th root
        lp_norm = sum(abs(item) ** p for item in flattened_data) ** (1 / p)

        return lp_norm

    def listeye(self):
        # Converts the gergen object into a list or a nested list, depending on its dimensions.
        if isinstance(self.__veri, list):
            if not self.__veri:
                return []
            return self.__veri.copy()
        else:
            return self.__veri

    def duzlestir(self):
        """Flattens a multidimensional list (self.__veri) into a 1D list."""
        if not isinstance(self.__veri, list):
            return gergen(self.__veri)
        flattened_list = []
        # Create a stack with the initial list
        stack = [self.__veri]

        # Process the stack
        while stack:
            current_item = stack.pop()
            if isinstance(current_item, list):
                # Extend the stack by reversing the current item list
                # to maintain the original order in the flattened list
                stack.extend(current_item[::-1])
            else:
                # If it's not a list, add it to the flattened list
                flattened_list.append(current_item)

        # Since we're appending elements to the end, but processing the stack in LIFO order,
        # we need to reverse the flattened list to restore the original element order
        # flattened_list.reverse()

        # Create a new gergen instance with the flattened list
        return gergen(flattened_list)

    def boyutlandir(self, yeni_boyut):
        """Reshapes the gergen object to a new shape 'yeni_boyut', specified as a tuple."""
        # Flatten the data first
        flat_data = list(self.duzlestir().__veri)

        def reshape_helper(data, dims):
            if not dims:
                return data.pop(0)
            return [reshape_helper(data, dims[1:]) for _ in range(dims[0])]

        # Check if the new shape is compatible with the number of elements
        if self.prod(yeni_boyut) != len(flat_data):
            raise ValueError(
                f"New shape must have the same number of elements as the original. {self.prod(yeni_boyut)} != {len(flat_data)}")

        # Use the helper to create the reshaped data and update the object's internal state
        self.__veri = reshape_helper(flat_data, yeni_boyut)
        self.__boyut = yeni_boyut
        return self

    def ic_carpim(self, other):
        ic_carpim_operation = IcCarpim()
        result_gergen = ic_carpim_operation(self, other)
        return result_gergen

    def dis_carpim(self, other):
        dis_carpim_operation = DisCarpim()
        result_gergen = dis_carpim_operation(self, other)
        return result_gergen

    def us(self, n):
        # Applies the power function to each element of the gergen object.
        power_operation = Us()
        result_gergen = power_operation(self, n)
        return result_gergen

    def log(self):
        # Applies the log function to each element of the gergen object.
        log_operation = Log10()
        result_gergen = log_operation(self)
        return result_gergen

    def ln(self):
        # Applies the ln function to each element of the gergen object.
        log_operation = Ln()
        result_gergen = log_operation(self)
        return result_gergen

    def sin(self):
        # Applies the sin function to each element of the gergen object.
        sin_operation = Sin()
        result_gergen = sin_operation(self)
        return result_gergen

    def cos(self):
        # Applies the cos function to each element of the gergen object.
        cos_operation = Cos()
        result_gergen = cos_operation(self)
        return result_gergen

    def tan(self):
        # Applies the tan function to each element of the gergen object.
        tan_operation = Tan()
        result_gergen = tan_operation(self)
        return result_gergen

    def topla(self, eksen=None):
        # Calculates the sum of the elements of the gergen object, optionally along a specified axis 'eksen'.
        topla_operation = Topla()
        result_gergen = topla_operation(self, eksen=eksen)
        return result_gergen

    def ortalama(self, eksen=None):
        # Calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
        ortalama_operation = Ortalama()
        result = ortalama_operation(self, eksen=eksen)
        result.operation = ortalama_operation
        return result

    def turev_al(self, grad_output=1):
        """
        TODO: Implement the backward pass for the gergen object
        """
        pass


# endregion Gergen Class

# endregion 1 The Gergen Library


# region 2 The MLP Implementation

# region 2.1 Katman Class
class Katman:
    def __init__(self, input_size: int, output_size: int, activation: Optional[str] = None,
                 init_method: Optional[str] = "xavier"):
        """
        Initializes the weights and biases of a neural network layer.
        Args:
            input_size (int): The number of input neurons
            output_size (int): The number of output neurons
            activation (str): The activation function to use. Default is None. ReLU and Softmax are supported.
            init_method (str): The weight initialization method to use. Default is "xavier". "xavier", "he", and "dogal" are supported.
        """

        """
        weights[j][i] represents the weight from the i-th input to the j-th output neuron.
        biases[j] represents the bias added to the j-th output neuron.

        Now its the reverse,
        weights[i][j] represents the weight from the i-th input to the j-th output neuron.
        """
        if init_method is None:
            self.weights = rastgele_gercek((input_size, output_size))
            self.biases = rastgele_gercek((output_size,))
        elif init_method.lower() == "xavier":
            # Xavier initialization, uses gaussian distribution with mean 0 and std_dev = sqrt(2 / (input_size + output_size))
            std_dev = (2 / (input_size + output_size)) ** 0.5
            mean = 0
            self.weights = rastgele_gercek((input_size, output_size), (mean, std_dev), "gaussian")
            self.biases = rastgele_gercek((output_size,), (mean, std_dev), "gaussian")
        elif init_method.lower() == "he":
            # He initialization, uses gaussian distribution with mean 0 and std_dev = sqrt(2 / input_size)
            std_dev = (2 / input_size) ** 0.5
            mean = 0
            self.weights = rastgele_gercek((input_size, output_size), (mean, std_dev), "gaussian")
            self.biases = rastgele_gercek((output_size,), (mean, std_dev), "gaussian")
        elif init_method.lower() == "dogal":
            # uses rastgele_dogal for testing purposes
            self.weights = rastgele_dogal((input_size, output_size), aralik=(-10, 10))
            self.biases = rastgele_dogal((1, output_size), aralik=(-10, 10))
        else:
            raise ValueError("Invalid initialization method.")

        if activation is None:
            self.activation = None
        elif activation.lower() == 'relu':
            self.activation = ReLU()
        elif activation.lower() == 'softmax':
            self.activation = Softmax()
        else:
            raise ValueError("Invalid activation function.")

    def __str__(self):
        """
        Returns a string representation of the layer.
        Returns:
            str: A string describing the layer
        """
        info = ""
        layer_info = f"Layer: {self.weights.boyut()[0]} -> {self.weights.boyut()[1]}"
        if self.activation is not None:
            activation_info = f"Activation: {self.activation.__class__.__name__}"
        else:
            activation_info = "Activation: None"

        weight_str = "\t" + "\n\t".join(self.weights.__str__().split("\n"))
        weights_info = f"Weights: \n{weight_str}"
        bias_str = "\t" + "\n\t".join(self.biases.__str__().split("\n"))
        biases_info = f"Biases: \n{bias_str}"

        info += f"{layer_info}\n{activation_info}\n{weights_info}\n{biases_info}"

        return info

    def ileri(self, x: gergen):
        """
        Forward pass of the layer.
        Args:
            x (gergen): Input to the layer
        Returns:
            gergen: Output of the layer, xW + b
        """
        result_of_matmul = x @ self.weights
        net = result_of_matmul + self.biases
        if len(net.boyut()) == 1:
            net = net.boyutlandir((1, net.uzunluk()))

        if self.activation is not None:
            result_of_act = self.activation(net)
            return result_of_act

        return net

    def reset_grads(self):
        self.weights.turev = None
        self.biases.turev = None

    def update_params(self, learning_rate: float):
        self.weights -= learning_rate * self.weights.turev
        self.weights.operation = None
        self.weights.requires_grad = False

        self.biases -= learning_rate * self.biases.turev
        self.biases.operation = None
        self.biases.requires_grad = False


# endregion 2.1 Katman Class

# region 2.2 ReLU Operation
class ReLU(Operation):
    def ileri(self, x: gergen) -> gergen:
        """
        The ReLU function modifies the input gergen by setting all its negative elements to zero while preserving the positive values.
        Args:
            x: gergen object
        Returns:
            gergen object: ReLU of the input gergen object
        """
        self.a = x
        self.b = None
        self.operands = [self.a]

        shape_x = x.boyut()
        elems = x.duzlestir().listeye()
        result = []
        for elem in elems:
            result.append(max(0, elem))

        return gergen(result, operation=self).boyutlandir(shape_x)

    def geri(self, grad_input=None):
        """
        The gradient of the ReLU function is 1 for positive values and 0 for others
        """
        if self.operands[0].operation is not None:
            grad = []
            for elem in self.a.duzlestir().listeye():
                grad.append(1 if elem > 0 else 0)

            # reverse because of lifo
            grad = grad[::-1]

            a_grad = gergen(grad).boyutlandir(self.a.boyut())

            if grad_input:
                self.operands[0].turev = grad_input * a_grad
            else:
                self.operands[0].turev = a_grad
            self.operands[0].turev.requires_grad = False

            self.operands[0].operation.geri(self.operands[0].turev)
        else:
            self.operands[0].turev = grad_input
            self.operands[0].turev.requires_grad = False
            # return self.operands[0].turev


# endregion 2.2 ReLU Operation

# region 2.3 Softmax Operation
class Softmax(Operation):
    def ileri(self, x: gergen) -> gergen:
        """
        Maps the input gergen object to a new gergen object using the softmax function.
        Args:
            x: gergen object
        Returns:
            gergen object: softmax of the input gergen object
        """
        self.a = x
        self.b = None
        self.operands = [self.a]
        shape_x = x.boyut()
        elems = x.duzlestir().listeye()

        # Compute the softmax function
        exps = [math.exp(elem) for elem in elems]
        sum_exps = sum(exps)
        result = [exp / sum_exps for exp in exps]

        result_gergen = gergen(result).boyutlandir(shape_x)
        result_gergen.operation = self

        return result_gergen

    def geri(self, grad_input):
        jacobian = []
        elems_of_a = self.a.duzlestir().listeye()
        for i, elem in enumerate(elems_of_a):
            row = []
            for j, elem2 in enumerate(elems_of_a):
                if i == j:
                    row.append(elem * (1 - elem))
                else:
                    row.append(-elem * elem2)
            jacobian.append(row)

        jacobian = gergen(jacobian).boyutlandir(self.a.boyut())

        if grad_input:
            self.operands[0].turev = grad_input @ jacobian
        else:
            self.operands[0].turev = jacobian

        self.operands[0].turev.requires_grad = False
        if self.operands[0].operation is not None:
            self.operands[0].operation.geri(self.operands[0].turev)


# endregion 2.3 Softmax Operation

# region 2.4 MLP Class
class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 activations: Optional[Tuple[str | None, str | None]] = ("relu", "softmax"),
                 init_method: Optional[str] = "xavier"):
        """
        Initializes the MLP with the given input, hidden, and output layer sizes.
        Args:
            input_size: The shape of input layer
            hidden_size: The shape of hidden layer
            output_size: How many outputs you need at the end, like how many categories you're classifying.
        """
        self.hidden_layer = Katman(input_size, hidden_size, activation=activations[0], init_method=init_method)
        self.output_layer = Katman(hidden_size, output_size, activation=activations[1], init_method=init_method)
        self.output = None

    def __str__(self):
        info = ""
        info += f"Layers: {self.hidden_layer.weights.boyut()[0]} -> {self.hidden_layer.activation.__class__.__name__}({self.hidden_layer.weights.boyut()[1]}) -> {self.output_layer.activation.__class__.__name__}({self.output_layer.weights.boyut()[1]})\n"
        return info

    def ileri(self, x: gergen):
        """
        Forward pass of the MLP.
        Args:
            x: gergen object
        Returns:
            gergen object: output of the MLP
        """
        hidden_layer_output = self.hidden_layer.ileri(x)
        output_layer_output = self.output_layer.ileri(hidden_layer_output)
        self.output = output_layer_output

        return self.output

    def calculate_gradients(self, y_true: gergen):
        """
        Backward pass of the MLP.
        Args:
            y_true: True labels
        Returns:
            gergen object: gradient of the loss with respect to the input
        """
        if self.output is None:
            raise ValueError("You need to run forward pass first.")

        # loss = cross_entropy(self.output, y_true)
        grad = cross_entropy_grad_wrt_logits(self.output, y_true)

        # Take the gradients of gergens recursively by taking the outputs geri
        self.output.operation.operands[0].operation.geri(grad)

    def reset_grads(self):
        self.hidden_layer.reset_grads()
        self.output_layer.reset_grads()

    def update_weights_and_biases(self, learning_rate: float):
        self.hidden_layer.update_params(learning_rate)
        self.output_layer.update_params(learning_rate)


# endregion 2.4 MLP Class

# region 2.5 Cross-Entropy Loss
def cross_entropy(y_pred: gergen, y_true: gergen) -> gergen:
    """
    Computes the cross-entropy loss between the predicted and true labels.
    Args:
        y_pred (gergen): Predicted labels
        y_true (gergen): True labels
    Returns:
        gergen: Cross-entropy loss
    """
    if not isinstance(y_pred, gergen):
        # y_pred is a list, convert it to a gergen object
        y_pred = gergen([y_pred])
    if not isinstance(y_true, gergen):
        # y_true is a list (one-hot encoded), convert it to a gergen object
        y_true = gergen([y_true])

    # Compute the cross-entropy loss using Gergen operations

    loss = (y_true * y_pred.log()).ortalama() * -1

    return loss


def cross_entropy_grad_wrt_logits(y_pred: gergen, y_true: gergen) -> gergen:
    """
    Computes the gradient of the cross-entropy loss with respect to the logits.
    This operation skips the gradient of softmax function.
    Args:
        y_pred (gergen): Predicted labels
        y_true (gergen): True labels
    Returns:
        gergen: Gradient of the cross-entropy loss with respect to the logits
    """
    if not isinstance(y_pred, gergen):
        # y_pred is a list, convert it to a gergen object
        y_pred = gergen([y_pred])
    if not isinstance(y_true, gergen):
        # y_true is a list (one-hot encoded), convert it to a gergen object
        y_true = gergen([y_true])

    # Compute the gradient of the cross-entropy loss with respect to the logits
    grad = y_pred - y_true

    grad.requires_grad = False

    return grad


# endregion 2.5 Cross-Entropy Loss

# endregion 2 The MLP Implementation

# region 2.6 Training Pipeline with egit()
def egit(mlp: MLP, inputs: gergen, targets: gergen, epochs: int, learning_rate: float, verbose: bool = True):
    """
    The `egit()` function adjusts the model's weights and biases to decrease errors
    and improve predictions through epochs.
    Args:
        mlp (MLP): The initialized MLP model
        inputs: The input data
        targets: The labels for each input
        epochs (int): The number of epochs to train the model
        learning_rate (float): The learning rate to update the weights
        verbose (bool): Whether to print the training progress
    """

    def print_progress_bar(iteration, total, prefix='', suffix='', length=50, elapsed_time=None):
        percent = "{0:.1f}".format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)

        if elapsed_time is not None:
            remaining_time = elapsed_time * (total - iteration) / iteration
            remaining_time_str = str(datetime.timedelta(seconds=int(remaining_time)))
        else:
            remaining_time_str = 'N/A'

        print(f'\r{prefix} |{bar}| {percent}% {suffix} - ETA: {remaining_time_str}', end='')
        if iteration == total:
            print()

    loss_history = []
    avg_loss_history = []
    total_sample_length = len(inputs.listeye())
    total_training_start_time = datetime.datetime.now()

    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_start_time = datetime.datetime.now()

        for index, sample in enumerate(inputs):
            if len(sample.boyut()) == 2 and sample.boyut()[0] == 1:
                single_sample = sample
                single_target = targets[index]
            else:
                single_sample = sample.boyutlandir((1, sample.boyut()[0]))
                single_target = targets[index].boyutlandir((1, targets[index].boyut()[0]))

            if single_sample.boyut()[1] != mlp.hidden_layer.weights.boyut()[0]:
                raise ValueError(
                    f"Input size does not match the input size of the model. Models input size is {mlp.hidden_layer.weights.boyut()[0]}, got {single_sample.boyut()[1]}")

            if single_target.boyut()[1] != mlp.output_layer.weights.boyut()[1]:
                raise ValueError(
                    f"Output size does not match the output size of the model. Models output size is {mlp.output_layer.weights.boyut()[1]}, got {single_target.boyut()[1]}")

            output = mlp.ileri(single_sample)

            loss = cross_entropy(output, single_target).listeye()

            mlp.calculate_gradients(single_target)

            # Update the weights and biases
            mlp.update_weights_and_biases(learning_rate)

            # Reset the gradients
            mlp.reset_grads()

            epoch_total_loss += loss
            loss_history.append(loss)

            elapsed_time = (datetime.datetime.now() - epoch_start_time).total_seconds()

            if verbose:
                print_progress_bar(index + 1, total_sample_length, prefix=f'Epoch {epoch + 1}/{epochs}',
                                   suffix='Complete', length=50, elapsed_time=elapsed_time)

        # Calculate and display epoch duration
        epoch_end_time = datetime.datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time

        # Append the loss to the history
        epoch_avg_loss = epoch_total_loss / total_sample_length
        avg_loss_history.append(epoch_avg_loss)

        if verbose:
            print(
                f"Epoch: {epoch + 1}, Total Loss: {epoch_total_loss}, Average Loss: {epoch_avg_loss}, Min Loss: {min(loss_history)}, Max Loss: {max(loss_history)}, Epoch Duration: {epoch_duration}\nHyperparameters: Learning Rate: {learning_rate}, Hidden Size: {mlp.hidden_layer.weights.boyut()[1]}")

    # Calculate and display total training duration
    total_training_end_time = datetime.datetime.now()
    total_training_duration = total_training_end_time - total_training_start_time

    if verbose:
        print(f"Total Training Duration for {epochs} epochs: {total_training_duration}")

    return mlp, loss_history, avg_loss_history


# endregion 2.6 Training Pipeline with egit()

# region 2.7 testing pipeline with test()
def test(mlp: MLP, inputs: gergen, targets: gergen):
    total_loss = 0
    total_sample_length = len(inputs.listeye())

    for index, sample in enumerate(inputs):
        if len(sample.boyut()) == 2 and sample.boyut()[0] == 1:
            single_sample = sample
            single_target = targets[index]
        else:
            single_sample = sample.boyutlandir((1, sample.boyut()[0]))
            single_target = targets[index].boyutlandir((1, targets[index].boyut()[0]))

        if single_sample.boyut()[1] != mlp.hidden_layer.weights.boyut()[0]:
            raise ValueError(
                f"Input size does not match the input size of the model. Models input size is {mlp.hidden_layer.weights.boyut()[0]}, got {single_sample.boyut()[1]}")

        if single_target.boyut()[1] != mlp.output_layer.weights.boyut()[1]:
            raise ValueError(
                f"Output size does not match the output size of the model. Models output size is {mlp.output_layer.weights.boyut()[1]}, got {single_target.boyut()[1]}")

        output = mlp.ileri(single_sample)

        loss = cross_entropy(output, single_target).listeye()

        total_loss += loss

    avg_loss = total_loss / total_sample_length

    print(f"Total Test Loss: {total_loss}, Average Test Loss: {avg_loss}")

    return avg_loss


# endregion 2.7 testing pipeline with test()

# region 2.8 Data Handling Process
def data_preprocessing(data_file, ratio=1.0, normalize=True):
    import pandas as pd
    import io
    from sklearn.preprocessing import LabelBinarizer

    # Load the data
    data = pd.read_csv(data_file)

    # take the ratio of the data
    if ratio < 1.0:
        data = data.sample(frac=ratio)

    # Get the first column as labels (You can use one-hot encoding if needed (You can use sklearn or pandas for this))
    lb = LabelBinarizer()
    labels = lb.fit_transform(data.iloc[:, 0])

    labels_list = []
    for label in labels:
        labels_list.append(label.tolist())

    # Get the remaining columns as data
    data = data.iloc[:, 1:].values

    data_list = []
    for row in data:
        data_list.append(row.tolist())

    if normalize:
        data_gergen = gergen(data_list) / 256
    else:
        data_gergen = gergen(data_list)

    data_gergen.requires_grad = False
    data_gergen.operation = None

    labels_gergen = gergen(labels_list)
    labels_gergen.requires_grad = False
    labels_gergen.operation = None

    # Return the data and labels
    return data_gergen, labels_gergen


# endregion 2.8 Data Handling Process

# region 2.9 Training and Testing the Model
def train_test_with_hyperparameters(train_file: str,
                                    test_file: str,
                                    learning_rates=None,
                                    hidden_sizes=None,
                                    epochs=None,
                                    ratio: float = 1.0,
                                    normalize: bool = True,
                                    verbose: bool = True):
    """
    Trains the MLP model with different hyperparameters. And finds the best hyperparameters.
    Args:
        train_file (str): The path to the data file
        test_file (str): The path to the test file
        learning_rates (list): The list of learning rates to try
        hidden_sizes (list): The list of hidden layer sizes to try
        epochs (list): The list of epochs to try
        ratio (float): The ratio of the data to use for training
        normalize (bool): Whether to normalize the data or not
        verbose (bool): Whether to print the training progress or not
    """
    if epochs is None:
        epochs = [10]
    if hidden_sizes is None:
        hidden_sizes = [5, 10, 30]
    if learning_rates is None:
        learning_rates = [0.01, 0.001, 0.001, 0.0001]

    # Load the data
    print(f"Loading the data from {train_file} and {test_file}...")
    train_inputs, train_targets = data_preprocessing(train_file, ratio=ratio, normalize=normalize)
    test_inputs, test_targets = data_preprocessing(test_file, ratio=ratio, normalize=normalize)

    # Train the model with different hyperparameters
    best_loss = float('inf')
    best_hyperparameters = None
    best_model = None
    best_avg_loss_history = None
    for lr in learning_rates:
        for hs in hidden_sizes:
            for ep in epochs:
                print(f"Training the model with Learning Rate: {lr}, Hidden Size: {hs}, Epochs: {ep}...")
                mlp = MLP(input_size=train_inputs.boyut()[1], hidden_size=hs, output_size=train_targets.boyut()[1])
                trained_model, loss_history, avg_loss_history = egit(mlp=mlp,
                                                                     inputs=train_inputs,
                                                                     targets=train_targets,
                                                                     epochs=ep,
                                                                     learning_rate=lr,
                                                                     verbose=verbose)
                avg_test_loss = test(trained_model, test_inputs, test_targets)
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    best_hyperparameters = (lr, hs, ep)
                    best_model = trained_model
                    best_avg_loss_history = avg_loss_history

    print(
        f"Best Hyperparameters:\n\tLearning Rate: {best_hyperparameters[0]},\n\tHidden Size: {best_hyperparameters[1]}")
    print(f"Best Average Test Loss: {best_loss}")

    # Plotting the loss curve
    plt.plot(best_avg_loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Loss Curve')
    plt.show()

    return best_model, best_loss, best_hyperparameters


# endregion 2.9 Training and Testing the Model

# region 3 Implementing in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim


# region 3.1 MLP_torch class

# Define the MLP architecture
class MLP_torch(nn.Module):
    """
    A simple MLP model with one hidden layer.
    hidden layers activation function: ReLU
    output layer activation function: Softmax
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_torch, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.hidden_activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden_activation(x)
        x = self.output(x)
        x = self.output_activation(x)
        return x


# endregion 3.1 MLP_torch class

# region 3.2 Data Preprocessing for MLP_Torch
def data_preprocessing_torch(data_file: str, ratio: float = 1.0, normalize: bool = True):
    import pandas as pd
    from sklearn.preprocessing import LabelBinarizer
    # Load the data
    data = pd.read_csv(data_file)

    # take the ratio of the data
    if ratio < 1.0:
        data = data.sample(frac=ratio)

    # Get the first column as labels (You can use one-hot encoding if needed (You can use sklearn or pandas for this))
    lb = LabelBinarizer()
    labels = lb.fit_transform(data.iloc[:, 0])

    # Get the remaining columns as data
    data = data.iloc[:, 1:].values

    # Normalize the data
    if normalize:
        data = data / 256

    # Convert the data and labels to torch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    return data, labels


# endregion 3.2 Data Preprocessing for MLP_Torch

# region 3.3 Training Pipeline for MLP_Torch
def train_torch(mlp: MLP_torch, inputs: torch.Tensor, targets: torch.Tensor, epochs: int, learning_rate: float):
    loss_history = []
    for epoch in range(epochs):
        """
        For each epoch:
            # Implement the training pipeline
            # Forward pass - with mlp.forward
            # Calculate Loss - with criterion (CrossEntropyLoss)
            # Backward pass - Compute gradients for example
            # Update parameters (Use an optimizer (e.g., SGD or Adam) to handle parameter updates)
        """

        # Forward pass
        output = mlp(inputs)

        # Calculate the loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, torch.argmax(targets, dim=1))
        loss_history.append(loss.item())

        # Backward pass
        optimizer = optim.SGD(mlp.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return mlp, loss_history


# endregion 3.3 Training Pipeline for MLP_Torch

# region 3.4 Testing Pipeline for MLP_Torch
def test_torch(mlp: MLP_torch, inputs: torch.Tensor, targets: torch.Tensor):
    """
    Test the model with the given inputs and targets.
    Args:
        mlp (MLP_torch): The trained MLP model
        inputs (torch.Tensor): The input data
        targets (torch.Tensor): The true labels
    Returns:
        float: The average loss
    """
    # Forward pass
    output = mlp(inputs)

    # Calculate the loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, torch.argmax(targets, dim=1))

    print(f'Test Loss: {loss.item()}')

    return loss.item()


# endregion 3.4 Testing Pipeline for MLP_Torch

# region 3.5 Training and Testing the Model with PyTorch
def train_test_with_hyperparameters_torch(train_file: str,
                                          test_file: str,
                                          learning_rates=None,
                                          hidden_sizes=None,
                                          epochs=None,
                                          ratio: float = 1.0,
                                          normalize: bool = False):
    """
    Trains the MLP model with different hyperparameters. And finds the best hyperparameters.
    Args:
        train_file (str): The path to the data file
        test_file (str): The path to the test file
        learning_rates (list): The list of learning rates to try
        hidden_sizes (list): The list of hidden layer sizes to try
        epochs (list): The list of epochs to try
        ratio (float): The ratio of the data to use for training
        normalize (bool): Whether to normalize the data or not
    """

    if epochs is None:
        epochs = [10]
    if hidden_sizes is None:
        hidden_sizes = [5, 10, 30]
    if learning_rates is None:
        learning_rates = [0.01, 0.001, 0.001, 0.0001]

    experiments_start_time = datetime.datetime.now()

    # Load the data
    print(f"Loading the data from {train_file} and {test_file}...")
    train_inputs, train_targets = data_preprocessing_torch(train_file, ratio=ratio, normalize=normalize)
    test_inputs, test_targets = data_preprocessing_torch(test_file, ratio=ratio, normalize=normalize)

    # Train the model with different hyperparameters
    best_loss = float('inf')
    best_hyperparameters = None
    best_model = None
    best_lost_history = None
    for lr in learning_rates:
        for hs in hidden_sizes:
            for ep in epochs:
                print(f"Training the model with Learning Rate: {lr}, Hidden Size: {hs}, Epochs: {ep}...")
                mlp = MLP_torch(input_size=train_inputs.shape[1], hidden_size=hs, output_size=train_targets.shape[1])
                trained_model, loss_history = train_torch(mlp=mlp,
                                                          inputs=train_inputs,
                                                          targets=train_targets,
                                                          epochs=ep,
                                                          learning_rate=lr)
                print(f"loss history:\n{loss_history}")
                avg_test_loss = test_torch(trained_model, test_inputs, test_targets)
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    best_hyperparameters = (lr, hs, ep)
                    best_model = trained_model
                    best_lost_history = loss_history

    print("\n\n")
    print("-" * 50)
    print(
        f"Best Hyperparameters:\n\tLearning Rate: {best_hyperparameters[0]},\n\tHidden Size: {best_hyperparameters[1]}")
    print(f"Best Average Test Loss: {best_loss}")
    print(f"Total Training Duration for All Experiments: {datetime.datetime.now() - experiments_start_time}")

    # Plotting the loss curve
    plt.plot(best_lost_history)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Loss Curve')
    plt.show()

    return best_model, best_loss, best_hyperparameters, best_lost_history
# endregion 3.5 Training and Testing the Model with PyTorch

# endregion 3 Implementing in PyTorch
