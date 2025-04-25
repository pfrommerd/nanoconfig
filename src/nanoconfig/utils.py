import itertools
import typing as ty
import collections.abc
import types
import requests
import contextlib

from rich.progress import Progress
from clearml.automation.optimization import Mapping

def flatten_items(data, prefix=""):
    for k, v in data.items():
        if prefix:
            k = f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from flatten_items(v, k)
        else:
            yield (k, v)

def flatten_dict(data, prefix=""):
    return dict(flatten_items(data, prefix=prefix))

def unflatten_dict(items : dict | ty.Iterator[tuple[str,ty.Any]]) -> dict:
    if isinstance(items, dict):
        items = items.items() # type: ignore
    result = {}
    for k, v in items:
        parts = k.split('.')
        parts = list(itertools.chain(*[part.split("/") for part in parts]))
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = v
    return result

def is_float(value: str):
    try:
        float(value)
        return True
    except ValueError:
        return False

def is_int(value: str):
    try:
        int(value)
        return True
    except ValueError:
        return False

def is_optional(t: object) -> bool:
    origin = ty.get_origin(t)
    if not (origin is ty.Union or origin is types.UnionType):
        return False
    a, b = ty.get_args(t)
    return a is type(None) or b is type(None)

def get_optional_type(t: object) -> ty.Optional[ty.Type]:
    origin = ty.get_origin(t)
    a, b = ty.get_args(t)
    return a if a is not None else b

_type = type

# Get list of comma-separated items,
# ignoring commas inside braces, brackets, or parenthesis
def split_list(value: str):
    level = 0
    idxs = [-1]
    for i, c in enumerate(value):
        if c in "[({":
            level += 1
        elif c in "])}":
            level -= 1
        elif c == "," and level == 0:
            idxs.append(i)
    idxs.append(len(value))
    return list(value[a+1:b] for a, b in itertools.pairwise(idxs))

def split_items(items: list[str]):
    idxs = [item.find(":") for item in items]
    keys = [item[:idx].strip() for item, idx in zip(items, idxs)]
    values = [item[idx+1:].strip() for item, idx in zip(items, idxs)]
    return zip(keys, values)

def parse_value(value: str, type: ty.Type) -> ty.Any:
    if is_optional(type):
        if value == "None" or value == "":
            return None
        type = get_optional_type(type) # type: ignore
        return parse_value(value, type)
    # If type is a tuple, list or dict,
    # convert it to tuple[ty.Any], list[ty.Any] or dict[str, ty.Any]
    if type in (tuple, list, ty.Sequence, ty.MutableSequence,
                collections.abc.Sequence, collections.abc.MutableSequence):
        type = type[ty.Any] # type: ignore
    elif type == dict:
        type = dict[str, ty.Any] # type: ignore
    # Check if we have a tuple, list or dict type
    if ty.get_origin(type) in (tuple, list,
                collections.abc.Sequence, collections.abc.MutableSequence,
                ty.Sequence, ty.MutableSequence):
        origin = ty.get_origin(type)
        if origin == ty.Sequence or origin == collections.abc.Sequence:
            origin = tuple
        elif origin == ty.MutableSequence or origin == collections.abc.MutableSequence:
            origin = list
        args = ty.get_args(type)
        if len(args) == 1:
            args = itertools.cycle([args[0]])
        value = value.strip()
        if not ((value[0] == "(" and value[-1] == ")")
                or (value[0] == "[" and value[-1] == "]")):
            raise ValueError(f"Invalid sequence format: {value}")
        items = split_list(value[1:-1])
        return origin(
            parse_value(item, arg)
            for item, arg in zip(items, args)
        ) # type: ignore
    elif ty.get_origin(type) in (dict, ty.Mapping,
                        collections.abc.Mapping,
                        collections.abc.MutableMapping):
        args = ty.get_args(type)
        key_type = args[0]
        value_type = args[1]
        value = value.strip()
        if value[0] != "{" or value[-1] != "}":
            raise ValueError(f"Invalid dict format: {value}")
        items = split_items(split_list(value[1:-1].strip()))
        return dict(
            (parse_value(k,key_type),parse_value(v,value_type))
            for k, v in items
        )
    if type == bool:
        return value.lower() == "true"
    elif type == _type(None):
        return None
    elif type in (int, float, str):
        return type(value)
    elif type == ty.Any:
        value = value.strip()
        if value == "None" or value == "":
            return parse_value(value, _type(None))
        elif value.lower() == "true" or value.lower() == "false":
            return parse_value(value, bool)
        elif is_int(value):
            return parse_value(value, int)
        elif is_float(value):
            return parse_value(value, float)
        elif ((value[0] == "[" and value[-1] == "]") or
              (value[0] == "(" and value[-1] == ")")):
            return parse_value(value, list[ty.Any])
        elif value[0] == "{" and value[-1] == "}":
            return parse_value(value, dict[str, ty.Any])
        return parse_value(value, str)
    else:
        raise ValueError(f"Unsupported type: {type}")

def download_url(path, url, job_name=None, quiet=False, pbar=None, response=None):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True) if response is None else response
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024*10 #10 Kibibyte
    if quiet or pbar is not None:
        ctx = contextlib.nullcontext()
    else:
        ctx = Progress()
        pbar = ctx
    with ctx:
        if pbar:
            task = pbar.add_task(
                f"{job_name}" if job_name is not None else "",
                total=total_size_in_bytes
            )
        with open(path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                if pbar:
                    pbar.update(task, advance=len(data)) # type: ignore
