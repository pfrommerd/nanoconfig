import pyarrow as pa
import numpy as np
import PIL.Image
import io

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

def as_arrow_array(array: np.ndarray) -> pa.Array:
    pa_array = pa.array(array.flatten())
    for s in array.shape[1:][::-1]:
        pa_array = pa.FixedSizeListArray.from_arrays(pa_array, s)
    return pa_array

# Will decode a pyarrow array with a given mime type to a numpy array
# this will handle e.g. image/any
def decode_as_numpy(array: pa.Array, field_mime_type: str | None = None) -> np.ndarray:
    if field_mime_type is None:
        return as_numpy(array)
    elif field_mime_type.startswith("image") and field_mime_type != "image/raw":
        # Decode all the images in the array
        images = []
        for image in array.field("bytes"):
            img = np.array(PIL.Image.open(io.BytesIO(image.as_py())))
            if img.ndim < 3: img = np.expand_dims(img, axis=-1)
            images.append(img.astype(np.float32)/255.0)
        return np.stack(images, axis=0)
    else:
        return as_numpy(array)
