import pyarrow as pa
import numpy as np

def as_numpy(array: pa.FixedSizeListArray) -> np.ndarray:
    shape = []
    type = array.type
    while pa.types.is_fixed_size_list(type):
        if type.list_size <= 0:
            raise ValueError("Invalid list size, jagged lists cannot be converted to numpy arrays.")
        shape.append(type.list_size)
        array = array.flatten()
        type = type.value_type
    return array.to_numpy(zero_copy_only=False).reshape(-1, *shape)

def as_array(array: np.ndarray) -> pa.Array:
    pa_array = pa.array(array.flatten())
    for s in array.shape[:-1:-1]:
        pa_array = pa.FixedSizeListArray.from_arrays(pa_array, s)
    return pa_array
