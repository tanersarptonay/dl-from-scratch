import random
import math
import re
import time
from typing import Union, Optional, Any
from itertools import product
import numpy as np  # For testing purposes

# Test functions and results after the Implementation part

"""
"""

"""
################## IMPLEMENTATION ##################
"""

"""
##### Helper Functions #####
"""


def is_nested_list(data: Any) -> bool:
    """
    Checks if the given data is a nested list.
    Args:
        data: The data to be checked.
    Returns:
        bool: True if the data is a nested list, False otherwise.
    """
    if not isinstance(data, Union[list, tuple]):
        return False
    if not any(isinstance(i, Union[list, tuple]) for i in data):
        return False
    return True


def depth_of_list(data: Union[int, float, list]) -> int:
    """
    Calculates the depth of a nested list.
    Args:
        data: The nested list to be checked.
    Returns:
        int: The depth of the nested list.
    """
    if isinstance(data, (int, float)):
        return 0
    if not is_nested_list(data):
        return 1
    return 1 + max(depth_of_list(i) for i in data)


def flatten(data: Any) -> list:
    """
    Flattens a nested list to a 1-dimensional list.
    Args:
        data: The nested list to be flattened.

    Returns: The flattened 1-dimensional list.
    """
    if not isinstance(data, list):
        return [data]
    else:
        return [elem for sublist in data for elem in flatten(sublist)]


def unflatten(flat_list: list, dimensions: Union[tuple, int]) -> list:
    """
    Reshape a 1-dimensional list to an N-dimensional list with respect to the given dimensions.
    Args:
        flat_list: The 1-dimensional list to be reshaped.
        dimensions: The dimensions of the reshaped list.

    Returns: The reshaped N-dimensional list.
    """
    # If dimensions has only one dimension, slice the flat list accordingly
    sum_dim = dimensions if isinstance(dimensions, int) else sum(dimensions)
    if sum_dim == 0:
        try:
            return flat_list[0]
        except IndexError:
            return flat_list

    if len(dimensions) == 1:
        return flat_list[:dimensions[0]]

    # Calculate the size of the next level of sublist
    stride = 1
    for dim in dimensions[1:]:
        stride *= dim

    # Recursively reshape the list
    unflattened_list = []
    for i in range(0, len(flat_list), stride):
        # For each subpart apply unflatten to get the next level of the unflattened list
        unflattened_list.append(unflatten(flat_list[i:i + stride], dimensions[1:]))

    return unflattened_list


"""
###### Operations ######
"""


class Operation:
    def __call__(self, *operands):
        """
        Makes an instance of the Operation class callable.
        Stores operands and initializes outputs to None.
        Invokes the forward pass of the operation with given operands.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the forward pass of the operation.
        """
        self.operands = operands
        self.outputs = None
        return self.ileri(*operands)

    def ileri(self, *operands):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError


# Comment lines of Operations are in gergen's operations

class Multiplication(Operation):
    def ileri(self, a: Union['gergen'], b: Union['gergen', int, float]):
        if isinstance(b, (int, float)) or (isinstance(b, gergen) and b.is_scalar()):
            # Scalar addition
            other_val = b if isinstance(b, (int, float)) else b.listeye()
            flat_veri = a.get_flat_veri()
            result = [element * other_val for element in flat_veri]
            new_veri = unflatten(result, a.boyut())
            return gergen(new_veri)
        elif isinstance(b, gergen):
            if a.is_scalar() and not b.is_scalar():
                return b * a

            if a.boyut() != b.boyut():
                raise ValueError(
                    f"Cannot multiply gergen objects with different shapes: {a.boyut()} and {b.boyut()}"
                )
            flat_self_veri = a.get_flat_veri()
            flat_other_veri = flatten(b.listeye())
            result = [s * o for s, o in zip(flat_self_veri, flat_other_veri)]
            new_veri = unflatten(result, a.boyut())
            return gergen(new_veri)
        else:
            raise TypeError(f"Unsupported operand type(s) for multiplication: 'gergen' and {type(b)}")


class TrueDivision(Operation):
    def ileri(self, a: Union['gergen'], b: Union['gergen', int, float]):
        if isinstance(b, (int, float)):
            # Scalar division
            if b == 0:
                raise ZeroDivisionError("Division by zero is not allowed")

            if a.is_scalar():
                return gergen(a.listeye() / b)

            flat_veri = a.get_flat_veri()
            result = [element / b for element in flat_veri]
            new_veri = unflatten(result, a.boyut())

            return gergen(new_veri)
        elif isinstance(b, gergen) and b.is_scalar():
            if b.listeye() == 0:
                raise ZeroDivisionError("Division by zero is not allowed")

            flat_veri = a.get_flat_veri()
            result = [element / b.listeye() for element in flat_veri]
            new_veri = unflatten(result, a.boyut())
            return gergen(new_veri)

        else:
            raise TypeError(f"Unsupported operand type(s) for division: 'gergen' and {type(b)}")


class RightTrueDivision(Operation):
    def ileri(self, a: Union['gergen', int, float], b: Union['gergen']):
        if isinstance(a, (int, float)):
            # Scalar division
            if a == 0:
                raise ZeroDivisionError("Division by zero is not allowed")

            if b.is_scalar():
                return gergen(a / b.listeye())

            flat_veri = b.get_flat_veri()
            result = [a / element for element in flat_veri]
            new_veri = unflatten(result, b.boyut())

            return gergen(new_veri)
        elif isinstance(a, gergen) and a.is_scalar():
            if a.listeye() == 0:
                raise ZeroDivisionError("Division by zero is not allowed")

            flat_veri = b.get_flat_veri()
            result = [a.listeye() / element for element in flat_veri]
            new_veri = unflatten(result, b.boyut())
            return gergen(new_veri)
        else:
            raise TypeError(f"Unsupported operand type(s) for division: {type(a)} and 'gergen'")


class Addition(Operation):
    def ileri(self, a: Union['gergen'], b: Union['gergen', int, float]):
        if isinstance(b, (int, float)) or (isinstance(b, gergen) and b.is_scalar()):
            # Scalar addition
            other_val = b if isinstance(b, (int, float)) else b.listeye()
            flat_veri = a.get_flat_veri()
            result = [element + other_val for element in flat_veri]
            new_veri = unflatten(result, a.boyut())
            return gergen(new_veri)
        elif isinstance(b, gergen):
            if a.is_scalar() and not b.is_scalar():
                return b + a
            if a.boyut() != b.boyut():
                raise ValueError(
                    f"Cannot add gergen objects with different shapes: {a.boyut()} and {b.boyut()}"
                )
            flat_self_veri = a.get_flat_veri()
            flat_other_veri = flatten(b.listeye())
            result = [s + o for s, o in zip(flat_self_veri, flat_other_veri)]
            new_veri = unflatten(result, a.boyut())
            return gergen(new_veri)
        else:
            raise TypeError(f"Unsupported operand type(s) for addition: 'gergen' and {type(b)}")


class Subtraction(Operation):
    def ileri(self, a: Union['gergen'], b: Union['gergen', int, float]):
        if isinstance(b, (int, float)) or (isinstance(b, gergen) and b.is_scalar()):
            # Scalar subtraction
            other_val = b if isinstance(b, (int, float)) else b.listeye()
            flat_veri = a.get_flat_veri()
            result = [element - other_val for element in flat_veri]
            new_veri = unflatten(result, a.boyut())
            return gergen(new_veri)
        elif isinstance(b, gergen):
            if a.is_scalar() and not b.is_scalar():
                return a + (-1 * b)
            if a.boyut() != b.boyut():
                raise ValueError(
                    f"Cannot subtract gergen objects with different shapes: {a.boyut()} and {b.boyut()}"
                )
            flat_self_veri = a.get_flat_veri()
            flat_other_veri = flatten(b.listeye())
            result = [s - o for s, o in zip(flat_self_veri, flat_other_veri)]
            new_veri = unflatten(result, a.boyut())
            return gergen(new_veri)
        else:
            raise TypeError(f"Unsupported operand type(s) for subtraction: 'gergen' and {type(b)}")


class Power(Operation):
    def ileri(self, a: Union['gergen'], n: int):
        if n < 0:
            raise ValueError(f"Invalid power value: {n}. Power should be a non-negative integer.")

        flat_veri = a.get_flat_veri()
        result = [element ** n for element in flat_veri]
        new_veri = unflatten(result, a.boyut())
        return gergen(new_veri)


class InnerProduct(Operation):
    def ileri(self, a: Union['gergen'], b: Union['gergen']):
        # from the ic_carpim function
        if not isinstance(b, gergen):
            raise TypeError(
                f"Unsupported operand type(s) for inner product: 'gergen' and {type(b)}"
            )

        elif len(a.boyut()) == 2 and len(b.boyut()) == 2 and a.boyut()[1] == b.boyut()[0]:
            # If both gergen objects are matrices, calculate the matrix multiplication
            result = [[0 for j in range(len(b.listeye()[0]))] for i in range(len(a.listeye()))]
            for i in range(len(a.listeye())):
                for j in range(len(b.listeye()[0])):
                    for k in range(len(b.listeye())):
                        result[i][j] += a.listeye()[i][k] * b.listeye()[k][j]

            return gergen(result)

        elif len(a.boyut()) == 1 and len(b.boyut()) == 1 and a.boyut() == b.boyut():
            # If both gergen objects are vectors, calculate the dot product
            flat_self_veri = a.listeye()
            flat_other_veri = b.listeye()
            result = sum(s * o for s, o in zip(flat_self_veri, flat_other_veri))
            return float(result)

        elif (a.boyut() == b.boyut()) and (
                (a.boyut()[0] == 1 and b.boyut()[1] == 1) or (a.boyut()[1] == 1 and b.boyut()[0] == 1)):
            # If one of the gergen objects is a vector, calculate the dot product
            flat_self_veri = a.get_flat_veri()
            flat_other_veri = b.get_flat_veri()
            result = sum(s * o for s, o in zip(flat_self_veri, flat_other_veri))
            return float(result)

        else:
            raise ValueError(
                f"Cannot calculate inner product of gergen objects with different shapes: {a.boyut()} and {b.boyut()}"
            )


class OuterProduct(Operation):
    def ileri(self, a: Union['gergen'], b: Union['gergen']):
        # Implements dis_carpim
        if not isinstance(b, gergen):
            raise TypeError(
                f"Unsupported operand type(s) for outer product: 'gergen' and {type(b)}"
            )

        if len(a.boyut()) > 2 or len(b.boyut()) > 2:
            raise ValueError(f"Both operands must be vectors to calculate outer product")

        if (1 not in a.boyut()) or (1 not in b.boyut()):
            if len(a.boyut()) != 1 or len(b.boyut()) != 1:
                raise ValueError(f"Both operands must be vectors to calculate outer product")

        new_matrix = []
        self_flat = a.get_flat_veri()
        other_flat = b.get_flat_veri()

        for i in range(a.uzunluk()):
            matrix_row = []
            for j in range(b.uzunluk()):
                matrix_row.append(self_flat[i] * other_flat[j])
            new_matrix.append(matrix_row)

        return gergen(new_matrix)


"""
##### Gergen Class #####
"""


class gergen:
    __veri = None  # A nested list of numbers representing the data
    D = None  # Transpose of data
    __boyut = None  # Dimensions of the derivative (Shape)
    __is_scalar = None
    __flat_veri = None
    __multiplication = Multiplication()
    __truedivision = TrueDivision()
    __righttruedivision = RightTrueDivision()
    __addition = Addition()
    __subtraction = Subtraction()
    __power = Power()
    __innerproduct = InnerProduct()
    __outerproduct = OuterProduct()

    def __init__(self, veri: Union[int, float, list, tuple, 'gergen', None] = None):
        """
        The constructor for the 'gergen' class.
        Args:
            veri: A nested list of numbers that represents the gergen data.
                The outer list contains rows, and each inner list contains the elements of each row.
                If 'veri' is None, the tensor is initialized without data.
        """
        if veri is None:
            self.__veri = []
            self.__is_scalar = False
        elif isinstance(veri, gergen):
            self.__veri = veri.listeye()
            self.__is_scalar = veri.is_scalar()
        elif isinstance(veri, (int, float)):
            self.__veri = veri
            self.__is_scalar = True
        elif isinstance(veri, (list, tuple)):
            self.__is_scalar = False
            if isinstance(veri, tuple):
                veri = list(veri)
            # list should be a nested list
            if all(isinstance(row, list) for row in veri):
                self.__veri = veri
            else:
                self.__veri = veri
        else:
            raise TypeError(f"Invalid data type for 'veri'. Expected int, float, list or gergen but got {type(veri)}")
        self.__boyut_guncelle()
        self.__devrik_guncelle()

    def get_flat_veri(self) -> list:
        """
        Returns the flattened representation of the gergen object.
        Returns:
            list: The flattened representation of the gergen object.
        """
        if self.__flat_veri is None:
            self.__flat_veri = flatten(self.__veri)
        return self.__flat_veri

    def __boyut_guncelle(self) -> None:
        """
        Updates the dimensions of the gergen object.
        Returns:
            None
        """
        if self.__veri is None or self.is_scalar():
            boyut_list = (0, 0)
        else:
            boyut_list = self.__boyut_guncelle_recursive(self.__veri)
        self.__boyut = tuple(boyut_list)

    def __boyut_guncelle_recursive(self, current_veri: Union[int, float, list]) -> list:
        """
        Recursively calculates the dimensions of the gergen object.
        Args:
            current_veri: Current part of the gergen object being processed.
        Returns:
            list: A list of integers representing the dimensions of the gergen object.
        """
        if isinstance(current_veri, (int, float)):
            return [1]
        elif isinstance(current_veri, list):
            boyut_list = [len(current_veri)]

            if len(current_veri) > 0 and is_nested_list(current_veri):
                boyut_list += self.__boyut_guncelle_recursive(current_veri[0])

            return boyut_list
        else:
            raise ValueError(f"Invalid data type in gergen: {type(current_veri)}")

    def __devrik_guncelle(self) -> None:
        """
        Updates the transpose of the gergen object.
        This method is called whenever the data of the gergen object is modified.
        """
        if not is_nested_list(self.__veri):
            self.D = self.__veri
            return

        flat_transpose = self.__get_flat_transpose(self.__veri)
        unflattened_transpose = unflatten(flat_transpose, self.boyut()[::-1])

        self.D = unflattened_transpose

    def __index_to_transposed_flat_index(self, indices: Union[list, tuple]) -> int:
        """
        Converts indices to the corresponding index in the flattened, transposed array.
        Args:
            indices: The indices to be converted.

        Returns:
            The index in the flattened, transposed array.
        """
        reverse_dimensions = self.boyut()[::-1]

        result_index = 0
        stride = 1  # Similar to pointer arithmetic, the stride is the number of elements to skip to move to the next index.

        # Iterate over each index and its corresponding dimension in reverse.
        for i, dim in zip(indices, reverse_dimensions):
            result_index += i * stride
            stride *= dim  # Update stride for the next dimension.

        return result_index

    def __transposed_indices(self) -> list:
        """
        Generates all possible combinations in order of indices for the transposed dimensions.

        Returns:
            A list of indices
        """
        flat_transpose_indices = []

        transposed_dimensions = self.boyut()[::-1]

        # Generate all possible combinations of indices for the given dimensions
        indices_product = product(*[range(d) for d in transposed_dimensions])

        # Print the index in the flattened, transposed array for each combination of indices
        for indices in indices_product:
            index = self.__index_to_transposed_flat_index(indices)
            flat_transpose_indices.append(index)

        return flat_transpose_indices

    def __get_flat_transpose(self, original_data: list) -> list:
        """
        Fills the values of the transposed, flattened array with respect to calculated indices.
        Args:
            original_data: The original data of the gergen object.

        Returns:
            list: The transposed, flattened array of the original data.
        """
        flat_transpose = []

        flat_original_data = flatten(original_data)

        trans_indeces = self.__transposed_indices()

        for t_index in trans_indeces:
            flat_transpose.append(flat_original_data[t_index])

        return flat_transpose

    def __getitem__(self, index: int) -> 'gergen':
        # Handling integers for indexing, returns a new gergen object
        if isinstance(index, int):
            if self.__veri is None:
                raise IndexError(f"Index out of range, gergen is empty")
            if index < 0:
                # Reverse accessing for negative indices
                index = self.boyut()[0] + index
            if index >= self.boyut()[0]:
                raise IndexError(
                    f"Index out of range, gergen has {self.boyut()[0]} rows, but index {index} was requested")

            result = self.__veri[index]

            if [result] == self.__veri:
                return gergen(result[index])
            else:
                return gergen(result)
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __str__(self) -> str:
        """
        Returns a string representation of the gergen object, similar to NumPy's array representation.
        Returns:
            A string representation of the gergen object.
        """
        # Generates a string representation
        if self.boyut() == (0, 0) or self.boyut() == (0,) or self.boyut() == 0:
            if self.is_scalar():
                return f"0 boyutlu skaler gergen:\n{self.__veri}\n"
            else:
                return f"0 boyutlu gergen:\n{self.__veri}\n"
        else:
            boyut_str = 'x'.join(str(i) for i in self.boyut()) + ' boyutlu gergen:\n'

            veri_str = str(self.__veri).replace("],", "]\n")
            veri_str = re.sub(r'\]{2,}', lambda match: match.group(0) + '\n', veri_str)

            return boyut_str + veri_str

    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Multiplication operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be multiplied with.

        Returns:
            The result of the multiplication.

        Raises:
            ValueError: If the gergen objects have different shapes.
            TypeError: If the multiplication is attempted with an unsupported type.
        """
        return self.__multiplication(self, other)

    def __rmul__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Right multiplication operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be multiplied with.

        Returns:
            The result of the multiplication.

        Raises:
            ValueError: If the gergen objects have different shapes.
            TypeError: If the multiplication is attempted with an unsupported type.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Division operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be divided by.

        Returns:
            The result of the division.
        Raises:
            ZeroDivisionError: If the scalar division is attempted with zero.
            TypeError: If the division is attempted with an unsupported type.
        """
        return self.__truedivision(self, other)

    def __rtruediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Right division operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be divided by.

        Returns:
            The result of the division.

        Raises:
            ZeroDivisionError: If the scalar division is attempted with zero.
            TypeError: If the division is attempted with an unsupported type.
        """
        return self.__righttruedivision(other, self)

    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Addition operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be added.

        Returns:
            The result of the addition.

        Raises:
            ValueError: If the gergen objects have different shapes.
            TypeError: If the addition is attempted with an unsupported type.
        """
        return self.__addition(self, other)

    def __radd__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Right addition operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be added.

        Returns:
            The result of the addition.

        Raises:
            ValueError: If the gergen objects have different shapes.
            TypeError: If the addition is attempted with an unsupported type.
        """
        return self.__add__(other)

    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Subtraction operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be subtracted.

        Returns:
            The result of the subtraction.

        Raises:
            ValueError: If the gergen objects have different shapes.
            TypeError: If the addition is attempted with an unsupported type.
        """
        return self.__subtraction(self, other)

    def __rsub__(self, other):
        """
        Right subtraction operation for gergen objects.
        Args:
            other: The other gergen object or scalar to be subtracted.

        Returns:
            The result of the subtraction.

        Raises:
            ValueError: If the gergen objects have different shapes.
            TypeError: If the addition is attempted with an unsupported type.
        """
        return -1 * self + other

    def uzunluk(self) -> int:
        """
        Returns the total number of elements in the gergen
        Returns:
            The total number of elements in the gergen
        """
        flat_veri = self.get_flat_veri()
        return len(flat_veri)

    def boyut(self) -> Union[int, tuple]:
        """
        Returns the dimensions of the gergen object.
        Returns:
            tuple: The dimensions of the gergen object.
        """
        if self.__boyut == (0, 0):
            return 0
        else:
            return self.__boyut

    def devrik(self) -> 'gergen':
        """
        Returns the transpose of the gergen object.
        Returns:
            gergen: The transpose of the gergen object.
        """
        devrik_g = gergen(self.D)
        return devrik_g

    def sin(self) -> 'gergen':
        """
        Applies the sine function to each of the elements in the gergen object.
        Returns:
            gergen: A new gergen object with the result of the sine function applied to each element.
        """
        flat_veri = self.get_flat_veri()
        result = [math.sin(element) for element in flat_veri]
        new_veri = unflatten(result, self.boyut())
        return gergen(new_veri)

    def cos(self) -> 'gergen':
        """
        Applies the cosine function to each of the elements in the gergen object.
        Returns:
            gergen: A new gergen object with the result of the cosine function applied to each element.
        """
        flat_veri = self.get_flat_veri()
        result = [math.cos(element) for element in flat_veri]
        new_veri = unflatten(result, self.boyut())
        return gergen(new_veri)

    def tan(self):
        """
        Applies the tangent function to each of the elements in the gergen object.
        Returns:
            gergen: A new gergen object with the result of the tangent function applied to each element.
        """
        flat_veri = self.get_flat_veri()
        result = [math.tan(element) for element in flat_veri]
        new_veri = unflatten(result, self.boyut())
        return gergen(new_veri)

    def us(self, n: int) -> 'gergen':
        """
        Raises each element of the gergen object to the power 'n'. This is an element-wise operation.
        Args:
            n: The power to raise each element to.

        Returns:
            gergen: A new gergen object with each element raised to the power 'n'.

        Raises:
            ValueError: If the power value is negative.
        """
        return self.__power(self, n)

    def log(self) -> 'gergen':
        """
        Applies the logarithm function to each element of the gergen object, using the base 10.
        Returns:
            gergen: A new gergen object with the result of the logarithm function applied to each element.
        """
        flat_veri = self.get_flat_veri()
        result = [math.log10(element) for element in flat_veri]
        new_veri = unflatten(result, self.boyut())
        return gergen(new_veri)

    def ln(self) -> 'gergen':
        """
        Applies the natural logarithm function to each element of the gergen object.
        Returns:
            gergen: A new gergen object with the result of the natural logarithm function applied to each element.
        """
        flat_veri = self.get_flat_veri()
        result = [math.log(element) for element in flat_veri]
        new_veri = unflatten(result, self.boyut())
        return gergen(new_veri)

    def L1(self) -> float:
        """
        Calculates the L1 norm. The L1 norm, also known as the Manhattan norm, is the sum of the absolute values of
        the elements in the tensor
        Returns:
            float: The L1 norm of the gergen object.
        """
        flat_veri = self.get_flat_veri()
        l1 = sum(abs(element) for element in flat_veri)
        return float(l1)

    def L2(self) -> float:
        """
        Calculates the L2 norm or the Euclidean norm, which is the square root of the sum of the squares of the
        tensor’s elements.

        Returns:
            float: The L2 norm of the gergen object.
        """
        flat_veri = self.get_flat_veri()
        l2 = math.sqrt(sum(element ** 2 for element in flat_veri))
        return float(l2)

    def Lp(self, p: int) -> float:
        """
        Calculates Lp norm, which is general version of L1 and L2 norms by calculating each element to the power of
        p, summing these values, and then taking the p-th root of the result.
        Args:
            p: The power to raise each element to.

        Returns:
            float: The Lp norm of the gergen object.

        Raises:
            ValueError: If the power value p is negative.
        """
        if p < 0:
            raise ValueError(f"Invalid power value: {p}. Power should be a non-negative integer.")

        flat_veri = self.get_flat_veri()
        lp = (sum(element ** p for element in flat_veri)) ** (1 / p)
        return float(lp)

    def listeye(self) -> Union[list, int, float, None]:
        """
        Returns the data of the gergen object as a list or a nested list, depending on its dimensions.
        Returns:
            list: The data of the gergen object.
        """
        return self.__veri

    def duzlestir(self) -> 'gergen':
        """
        Converts the gergen object's multidimensional structure into a 1D structure, effectively 'flattening' the object.
        Returns:
            list: The flattened representation of the gergen object.
        """
        return gergen(self.get_flat_veri())

    def boyutlandir(self, yeni_boyut: tuple) -> 'gergen':
        """
        Reshapes the gergen object to a new shape 'yeni_boyut'
        Args:
            yeni_boyut: The new shape to reshape the gergen object to.

        Returns:
            gergen: A new gergen object with the reshaped data.

        Raises:
            TypeError: If the type of 'yeni_boyut' is not a tuple.
            ValueError: If the new shape 'yeni_boyut' has a different number of elements than the original gergen object.
        """
        # Reshapes the gergen object to a new shape 'yeni_boyut', which is specified as a tuple.
        if not isinstance(yeni_boyut, tuple):
            raise TypeError(f"Invalid type for 'yeni_boyut'. Expected tuple, but got {type(yeni_boyut)}")

        # multiplication of yeni_boyut
        new_element_number = 1
        for i in yeni_boyut:
            new_element_number *= i

        old_element_number = self.uzunluk()

        if new_element_number != old_element_number:
            raise ValueError(f"Cannot reshape gergen of size {old_element_number} into shape {yeni_boyut}")

        flat_veri = self.get_flat_veri()
        new_veri = unflatten(flat_veri, yeni_boyut)

        return gergen(new_veri)

    def ic_carpim(self, other: 'gergen') -> Union[float, 'gergen']:
        """
        Calculates the inner (dot) product of this gergen object with another.
        Args:
            other: The other gergen object to calculate the inner product with.

        Returns:
            The result of the inner product.

        Raises:
            TypeError: If the inner product is attempted with an unsupported type.
            ValueError: If the gergen objects have different dimensions.
        """
        return self.__innerproduct(self, other)

    def dis_carpim(self, other: 'gergen') -> 'gergen':
        """
        Calculates the outer product of this gergen object with another.
        Args:
            other: The other vector gergen to calculate the outer product with.

        Returns:
            The result of the outer product, which is a matrix.

        Raises:
            TypeError: If the outer product is attempted with an unsupported type.
            ValueError: If the gergen objects have different dimensions.
        """
        return self.__outerproduct(self, other)

    def topla(self, eksen: Optional[int] = None) -> Union['gergen', float]:
        """
        Sums up the elements of the gergen object, optionally along a specified axis 'eksen'.
        Args:
            eksen: The axis to sum the elements along. If None, the sum of all elements is calculated.

        Returns:
            The result of the summation.

        Raises:
            TypeError: If the type of 'eksen' is not an integer or None.
            ValueError: If the axis is out of bounds for the gergen object.
        """
        if not isinstance(eksen, int) and eksen is not None:
            raise TypeError(f"Invalid type for 'eksen'. Expected int or None, but got {type(eksen)}")

        if eksen is not None and (eksen < 0 or eksen >= len(self.boyut())):
            raise ValueError(f"Axis out of bounds: {eksen}. The gergen object has {len(self.boyut())} dimensions")

        if eksen is None:
            flat_veri = self.get_flat_veri()
            return gergen(sum(flat_veri))

        new_veri = self.__topla_recursive(self.__veri, eksen)

        return gergen(new_veri)

    def __topla_recursive(self, data: list, eksen: int, depth: int = 0):
        """
        Recursively calculates the sum of the elements of the gergen object, optionally along a specified axis 'eksen'.
        Args:
            data: The data to be summed.
            eksen: The axis to sum the elements along.
            depth: The current depth of the recursion.

        Returns:
            The result of the summation.
        """
        if not data:
            return 0 if depth == eksen else []

        if depth == eksen:
            if is_nested_list(data[0]):
                # for handling the case when eksen is 0
                return [self.__topla_recursive(list(sub_arr), eksen, depth) for sub_arr in zip(*data)]
            elif is_nested_list(data):
                return [sum(pair) for pair in zip(*data)]
            elif isinstance(data, list):
                return sum(data)
            else:
                return data
        else:
            return [self.__topla_recursive(sublist, eksen, depth + 1) for sublist in data]

    def ortalama(self, eksen: Optional[int] = None) -> Union['gergen', float]:
        # Calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
        if not isinstance(eksen, int) and eksen is not None:
            raise TypeError(f"Invalid type for 'eksen'. Expected int or None, but got {type(eksen)}")

        if eksen is not None and (eksen < 0 or eksen >= len(self.boyut())):
            raise ValueError(f"Axis out of bounds: {eksen}. The gergen object has {len(self.boyut())} dimensions")

        if eksen is None:
            flat_veri = self.get_flat_veri()
            return sum(flat_veri) / len(flat_veri)

        new_veri = self.__ortalama_recursive(self.__veri, eksen)

        return gergen(new_veri)

    def __ortalama_recursive(self, data: list, eksen: int, depth: int = 0) -> Union[list, float]:
        """
        Recursively calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
        Args:
            data: The data to be averaged.
            eksen: The axis to average the elements along.
            depth: The current depth of the recursion.

        Returns:
            The result of the averaging.
        """
        if not data:
            return 0 if depth == eksen else []

        if depth == eksen:
            if is_nested_list(data[0]):
                # for handling the case when eksen is 0
                return [self.__ortalama_recursive(list(sub_arr), eksen, depth) for sub_arr in zip(*data)]
            elif is_nested_list(data):
                return [sum(pair) / len(pair) for pair in zip(*data)]
            elif isinstance(data, list):
                return sum(data) / len(data)
            else:
                return data
        else:
            return [self.__ortalama_recursive(sublist, eksen, depth + 1) for sublist in data]

    def is_scalar(self) -> bool:
        """
        Checks if the gergen object is a scalar.
        Returns:
            bool: True if the gergen object is a scalar, False otherwise.
        """
        return self.__is_scalar


"""
##### Random Data Generation #####
"""


def cekirdek(sayi: int) -> None:
    # Sets the seed for random number generation
    random.seed(sayi)


def rastgele_dogal(boyut: tuple, aralik: Optional[tuple] = (0, 100), dagilim: Optional[str] = 'uniform'):
    """
    Generates data of specified dimensions with random integer values and returns a gergen object.

    Args:
        boyut: Shape of the desired data.
        aralik: (min, max) specifying the range of random values. Defaults to (0,100), which implies a default range.
        dagilim: Distribution of random values ('uniform'). Defaults to 'uniform'.

    Returns:
        gergen: A new gergen object with random integer values.

    Raises:
        ValueError: If the distribution is not 'uniform' or None.
    """
    if dagilim != 'uniform':
        raise ValueError(f"Invalid distribution: {dagilim}. Only 'uniform' distribution is supported.")

    if not boyut:
        return gergen(random.randint(aralik[0], aralik[1]))

    # element_number is the multiplication of boyut
    element_number = 1
    for i in boyut:
        element_number *= i

    # Generate random integer values
    result = [random.randint(aralik[0], aralik[1]) for _ in range(element_number)]
    return gergen(unflatten(result, boyut))


def rastgele_gercek(boyut: tuple, aralik: Optional[tuple] = (0.0, 1.0), dagilim: Optional[str] = 'uniform'):
    """
    Generates a gergen of specified dimensions with random floating-point values.

    Args:
        boyut (tuple): Shape of the desired gergen.
        aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0.0, 1.0) for uniform distribution.
        dagilim (string, optional): Distribution of random value ('uniform'). Defaults to 'uniform'.

    Returns:
        gergen: A new gergen object with random floating-point values.

    Raises:
        ValueError: If the distribution is not 'uniform' or None.
    """
    if dagilim != 'uniform':
        raise ValueError(f"Invalid distribution: {dagilim}. Only 'uniform' distribution is supported.")

    if not boyut:
        return gergen(random.uniform(aralik[0], aralik[1]))

    # element_number is the multiplication of boyut
    element_number = 1
    for i in boyut:
        element_number *= i

    # Generate random floating-point values
    result = [random.uniform(aralik[0], aralik[1]) for _ in range(element_number)]

    return gergen(unflatten(result, boyut))


"""
##################### TESTS #####################
"""

"""
Test Results:

    example 1:
        Time taken for gergen: 0.08179879188537598
        Time taken for numpy : 3.9577484130859375e-05
        Time difference: 0.08175921440124512
        Time ratio: 2066.8012048192772
        Same results
        
    example 2:
        Time taken for gergen: 0.14371538162231445
        Time taken for numpy : 0.00013399124145507812
        Time difference: 0.14358139038085938
        Time ratio: 1072.5729537366549
        Same results
    
    example 3:
        Time taken for gergen: 0.10491299629211426
        Time taken for numpy : 0.0003497600555419922
        Time difference: 0.10456323623657227
        Time ratio: 299.95705521472394
        Same results
"""


def is_different(g: gergen, n):
    epsilon = 1e-10
    g_flat = flatten(g.listeye() if isinstance(g, gergen) else g)
    n_flat = flatten(n.tolist() if isinstance(n, np.ndarray) else n)
    for i in range(len(g_flat)):
        if abs(g_flat[i] - n_flat[i]) > epsilon:
            print("Different results")
    print("Same results")


def example_1():
    # Example 1
    test_boyut = (64, 64)
    g1 = rastgele_gercek(test_boyut)
    g2 = rastgele_gercek(test_boyut)

    np_g1 = np.array(g1.listeye())
    np_g2 = np.array(g2.listeye()).transpose()

    start = time.time()
    result_gergen = g1.ic_carpim(g2)
    end = time.time()

    start_np = time.time()
    # Apply the same equation for NumPy equivalent
    result_np = np.inner(np_g1, np_g2)
    end_np = time.time()

    print("Time taken for gergen:", end - start)
    print("Time taken for numpy :", end_np - start_np)
    print("Time difference:", (end - start) - (end_np - start_np))
    print("Time ratio:", (end - start) / (end_np - start_np))
    is_different(result_gergen, result_np)


def example_2():
    test_boyut = (4, 16, 16, 16)
    a = rastgele_gercek(test_boyut)
    b = rastgele_gercek(test_boyut)
    c = rastgele_gercek(test_boyut)

    np_a = np.array(a.listeye())
    np_b = np.array(b.listeye())
    np_c = np.array(c.listeye())

    start = time.time()
    result = (a * b + a * c + b * c).ortalama()
    end = time.time()

    start_np = time.time()
    result_np = (np_a * np_b + np_a * np_c + np_b * np_c).mean()
    end_np = time.time()

    print("Time taken for gergen:", end - start)
    print("Time taken for numpy :", end_np - start_np)
    print("Time difference:", (end - start) - (end_np - start_np))
    print("Time ratio:", (end - start) / (end_np - start_np))
    is_different(result, result_np)


def example_3():
    test_boyut = (3, 64, 64)
    a = rastgele_gercek(test_boyut)
    b = rastgele_gercek(test_boyut)

    np_a = np.array(a.listeye())
    np_b = np.array(b.listeye())

    start = time.time()
    result = (a.sin() + b.cos().us(2)).ln() / 8
    end = time.time()

    start_np = time.time()
    result_np = np.log((np.sin(np_a) + np.cos(np_b) ** 2)) / 8
    end_np = time.time()

    print("Time taken for gergen:", end - start)
    print("Time taken for numpy :", end_np - start_np)
    print("Time difference:", (end - start) - (end_np - start_np))
    print("Time ratio:", (end - start) / (end_np - start_np))
    is_different(result, result_np)
