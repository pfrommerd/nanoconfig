from dataclasses import (
    dataclass, field as _field,
    fields,
    MISSING as DC_MISSING,
)
from . import utils
import typing as ty
import functools
import abc

T = ty.TypeVar("T")

class Missing:
    def __repr__(self):
        return "???"

MISSING : ty.Any = Missing() # type: ignore

class ConfigType(type):
    def __init__(self, cls, bases, namespace):
        super().__init__(cls, bases, namespace)
        self.__variants__ = {}

# For abstract classes, we need to use a metaclass
# that is a subclass of ConfigType, but also abc.ABCMeta
class AbstractConfigType(ConfigType, abc.ABCMeta):
    pass

class Config(object, metaclass=ConfigType):
    def to_dict(self) -> dict[str, ty.Any]:
        res = {}
        for f in fields(self): # type: ignore
            v = getattr(self, f.name)
            if isinstance(v, Config):
                res[f.name] = v.to_dict()
                v_t = type(v)
                if f.type != v_t:
                    if not hasattr(f.type, "__variants__"):
                        raise TypeError(
                            f"Field {f.name} has type {f.type}, but value is {type(v)}"
                        )
                    type_variants = {v:k for k,v in f.type.__variants__.items()} # type: ignore
                    if not v_t in type_variants:
                        raise TypeError(
                            f"Field {f.name} has type {f.type}, but value is {type(v)}"
                        )
                    res[f.name]["type"] = type_variants[v_t]
            else:
                res[f.name] = v
        return res

    @classmethod
    def from_dict(cls, data: dict[str, ty.Any]) -> ty.Self:
        # Create a new instance of the class
        args = {}
        for f in fields(cls): # type: ignore
            if isinstance(f.type, ty.Type) and issubclass(f.type, Config):
                sub_dict = data.get(f.name, None)
                if sub_dict is None:
                    if f.default is not DC_MISSING:
                        args[f.name] = f.default
                else:
                    if "type" in sub_dict: # type: ignore
                        args[f.name] = f.type.__variants__[sub_dict["type"]].from_dict(sub_dict) # type: ignore
                    else:
                        args[f.name] = f.type.from_dict(sub_dict)
            else:
                value = data.get(f.name, f.default)
                if isinstance(value, str):
                    value = utils.parse_value(value, f.type) # type: ignore
                args[f.name] = value
        instance = cls(**args)
        return instance

def field(*, default: ty.Any = MISSING,
          default_factory: ty.Callable[[], ty.Any] | Missing = MISSING,
          flat: bool = False) -> ty.Any:
    # Add the field
    return _field(
        default=DC_MISSING if default is MISSING else default,
        default_factory=DC_MISSING if default_factory is MISSING else default_factory, # type: ignore
        metadata={"flat": flat}) # type: ignore

@ty.dataclass_transform()
@ty.overload
def config(cls: None = None, *, variant: str | None = None) -> ty.Callable[[ty.Type[T]], ty.Type[T]]: ...

@ty.dataclass_transform()
@ty.overload
def config(cls: ty.Type[T], *, variant: str | None = None) -> ty.Type[T]: ...

def config(cls = None, *, variant = None): # type: ignore
    if cls is None:
        return functools.partial(config, variant=variant)
    # Handle abstract classes
    if abc.ABC in cls.mro():
        metaclass = AbstractConfigType
    else:
        metaclass = ConfigType

    cls = dataclass(cls, frozen=True) # type: ignore
    class clz(cls, Config, metaclass=metaclass):
        pass
    # Mixin Config
    clz.__name__ = cls.__name__
    clz.__qualname__ = cls.__qualname__
    clz.__module__ = cls.__module__
    clz.__doc__ = cls.__doc__

    for f in cls.__dataclass_fields__.values():
        if isinstance(f.type, str):
            raise TypeError(
                f"Field type specifier for {f.name} is not a type, got {f.type} "
            )

    # Add as a variant to all Config classes in mro.
    if variant is not None:
        for base_cls in clz.mro()[:-1]:
            if isinstance(base_cls, ConfigType):
                base_cls.__variants__[variant] = clz
    return clz
