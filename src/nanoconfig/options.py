from . import Config, Missing, MISSING, utils

from dataclasses import dataclass, fields, MISSING as DC_MISSING
import typing as ty
import types
import abc
import sys
import logging
logger = logging.getLogger(__name__)

_type = type
T = ty.TypeVar("T")

class OptionParseError(ValueError):
    pass

@dataclass
class Option:
    name: str
    # The following have no practical impact
    # on the parsing part, but are used for help documentation

    # May be int, float, str, bool,
    # dict[str, T], list[T], tuple[T]
    # ty.Any yields a generic type of one of the above.
    type: type
    # A default value. Used for help documentation,
    # but not actually returned as a default value.
    # If MISSING, then the option is required.
    default: ty.Any | Missing

@dataclass
class Options(ty.Generic[T]):
    root_type : ty.Type[T]
    default : ty.Any | Missing
    opts : list[Option]

    @staticmethod
    def as_options(type: type[T],
                   default: T | Missing = MISSING,
                   *, prefix : str = "") -> "Options[T]":
        return Options(
            type, default,
            list(_as_options(type, default=default, prefix=prefix))
        )

    def parse(self,
        args: list[str] | None = None,
        parse_all: bool = True,
        parse_help = True
    ) -> dict[str, str]:
        if args is None:
            args = list(sys.argv[1:])
        return _parse_cli_options(self.opts, args=args,
            parse_all=parse_all, parse_help=parse_help)

    def from_parsed(self,
        options: dict[str, str],
    ) -> T:
        return _from_parsed_options(options, self.root_type,
            default=self.default, prefix=""
        ) # type: ignore

def _as_options(type : ty.Type[T], default : T | Missing = MISSING, *,
                # For variants, the type of the base class for the variant.
                # We will not include options contained in the base type.
                relative_to_base : ty.Type | None = None,
                prefix : str = "",
            ) -> ty.Iterator[Option]:
    if isinstance(type, ty.Type) and issubclass(type, Config):
        # If we have variants, add a "type" option
        if _has_variant_tag(type, relative_to_base): # type: ignore
            variants = type.__variants__
            variant_lookup = {type: alias for alias, type in variants.items()}
            # If the type is an abstract class,
            # we require the user to specify
            type_default = MISSING if isinstance(type, abc.ABCMeta) else type
            if default is not MISSING:
                type_default = _type(default)
            if type_default is not MISSING and type_default not in variant_lookup:
                raise ValueError(f"No variant registration for type {type_default} at \"{prefix}\".")
            type_default = (variant_lookup[type_default]
                            if type_default is not MISSING else MISSING)
            yield Option(_join(prefix, "type"), str, type_default)
        # Do not output fields for the base class
        # if relative_to_base is not None
        base_fields = (
            set() if relative_to_base is None else
            set(f.name for f in fields(relative_to_base))
        )
        for f in fields(type): # type: ignore
            if f.name in base_fields: # We should not include these fields
                continue
            field_default = (
                getattr(default, f.name) if default is not MISSING else
                (f.default if f.default is not DC_MISSING else MISSING)
            )
            flat = f.metadata.get("flat", False)
            yield from _as_options(f.type, default=field_default, # type: ignore
                               prefix=prefix if flat else _join(prefix, f.name))

        # If we have variants, output each of the variant options
        if _has_variant_tag(type, relative_to_base): # type: ignore
            for alias, variant_type in type.__variants__.items():
                # If the type is the same as the base type,
                # we do not need to output the base type
                if variant_type == type:
                    continue
                # Only specify the default if the type is the same.
                subvariant_default = (
                    default if variant_type is _type(default) else MISSING
                )
                yield from _as_options(variant_type,
                    default=subvariant_default,
                    prefix=_join(prefix, alias),
                    relative_to_base=type
                )
    else:
        yield Option(prefix, type, default)

def _has_variant_tag(type: ty.Type[T], relative_to_base: ty.Type):
    if relative_to_base is not None:
        return False
    elif type.__variants__:
        variant_lookup = {type: alias for alias, type in type.__variants__.items()}
        return not (len(variant_lookup) == 1 and variant_lookup.get(type, False))

# Will parse the options, removing any parsed
# options from the dictionary
def _from_parsed_options(options: dict[str, str],
                  type : ty.Type[T], default: T | Missing = MISSING, *,
                  prefix="") -> T | Missing:
    if isinstance(type, ty.Type) and issubclass(type, Config):
        if type.__variants__:
            variant_lookup = {t: alias for alias, t in type.__variants__.items()}
            default_type = _type(default) if default is not MISSING else (
                type if not isinstance(type, abc.ABCMeta) else MISSING
            )
            default_variant = variant_lookup.get(default_type, MISSING)
            variant = options.pop(_join(prefix, "type"), default_variant)
            if variant not in type.__variants__:
                raise OptionParseError(f"Invalid variant {variant} for {type}")
            config_type = type.__variants__[variant]
            # If we specified a variant different than the default, remove the default value.
            if config_type != default_type: default = MISSING
        else:
            if default is not MISSING and type != _type(default):
                raise OptionParseError(f"Default type and specified type must match exactly.")
            config_type = type
            variant = MISSING
        config_fields = {}
        # First go through the base fields
        for f in fields(type): # type: ignore
            flat = f.metadata.get("flat", False)
            config_fields[f.name] = (
                prefix if flat else _join(prefix, f.name), f
            )
        # Go through the variant-specific fields
        if config_type is not type:
            for f in fields(config_type): # type: ignore
                # If we are overriding a base field,
                # override the field so we get the updated default
                flat = f.metadata.get("flat", False)
                if f.name in config_fields:
                    config_fields[f.name] = (
                        prefix if flat else _join(prefix, f.name), f
                    )
                else:
                    assert variant is not MISSING, "variant must be set"
                    config_fields[f.name] = (
                        _join(prefix, variant) if flat else
                        _join(prefix, f"{variant}.{f.name}"), f
                    )
        # Now we can parse the options,
        config_args = {}
        for field_prefix, f in config_fields.values():
            default_value = (
                getattr(default, f.name) if default is not MISSING else
                (f.default if f.default is not DC_MISSING else MISSING)
            )
            value = _from_parsed_options(options, f.type, default_value,
                                        prefix=field_prefix)
            if value is not MISSING:
                config_args[f.name] = value
        return config_type(**config_args) # type: ignore
    else:
        opt : str | Missing = options.get(prefix, MISSING)
        if opt is not MISSING:
            return utils.parse_value(opt, type) # type: ignore
        else:
            return default

def _parse_cli_options(options: ty.Iterable[Option],
                  args: list[str],
                  parse_all: bool = True, parse_help = True) -> dict[str, str]:
    options = list(options)
    options.append(Option("help", bool, False))
    valid_keys = set(o.name for o in options)
    parsed_options = {}
    parsed_args = []
    last_key = None
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            last_key = None
            arg = arg[2:]
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in valid_keys:
                    parsed_args.append(i)
                    parsed_options[key] = value
            else:
                last_key = arg
                # For --flag, we set the value to "true"
                if last_key in valid_keys:
                    parsed_args.append(i)
                    parsed_options[last_key] = "true"
        elif last_key is not None:
            if last_key in valid_keys:
                parsed_args.append(i)
                parsed_options[last_key] = arg
            last_key = None
    # Remove from the list
    for i in reversed(parsed_args):
        del args[i]
    if parse_all and len(args) > 0:
        raise OptionParseError(f"Unknown options {args}. Valid options are {valid_keys}.")
    for opt in options:
        if opt.default is MISSING and opt.name not in parsed_options:
            raise OptionParseError(f"Missing option {opt.name} for {opt.type}")

    if parsed_options.get("help", False):
        print("Options:")
        for opt in options:
            type_name = (opt.type.__qualname__
                if isinstance(opt.type, ty.Type) else
                str(opt.type)
            )
            if opt.type == bool:
                print(f"  --{opt.name}")
            else:
                if opt.default is not MISSING:
                    print(f"  --{opt.name}={opt.default} ({type_name})")
                else:
                    print(f"  --{opt.name} ({type_name})")
        sys.exit(0)
    return parsed_options

def _join(prefix, name):
    if prefix:
        return f"{prefix}.{name}"
    return name
