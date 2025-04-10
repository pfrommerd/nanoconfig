from dataclasses import (
    dataclass, field as _field,
    MISSING as DC_MISSING,
)
import typing as ty
import functools
import abc

T = ty.TypeVar("T")

class Missing:
    def __repr__(self):
        return "???"

MISSING = Missing()

class ConfigType(type):
    def __init__(self, cls, bases, namespace):
        super().__init__(cls, bases, namespace)
        self.__variants__ = {}

# For abstract classes, we need to use a metaclass
# that is a subclass of ConfigType, but also abc.ABCMeta
class AbstractConfigType(ConfigType, abc.ABCMeta):
    pass

class Config(object, metaclass=ConfigType):
    pass

def field(*, default: ty.Any = MISSING, 
          default_factory: ty.Callable[[], ty.Any] | Missing = MISSING,
          flat: bool = False) -> ty.Any:
    # Add the field
    return _field(
        default=DC_MISSING if default is MISSING else default,
        default_factory=DC_MISSING if default_factory is MISSING else default_factory,
        metadata={"flat": flat})

@ty.dataclass_transform(field_specifiers=(field,))
def config(cls: ty.Type[T] = None, *, variant : str | None = None) -> type:
    if cls is None:
        return functools.partial(config, variant=variant)
    # Handle abstract classes
    if abc.ABC in cls.mro():
        metaclass = AbstractConfigType
    else:
        metaclass = ConfigType

    cls = dataclass(cls, frozen=True)
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