from nanoconfig import config, MISSING
from nanoconfig.options import Options

@config
class Simple:
    a: int = 1
    b: str = MISSING
    c: float = 1.0

def test_simple():
    config_opts = Options.as_options(Simple)
    parsed = config_opts.parse(["--b=b","--a=2","--c=3.0"])
    assert parsed["a"] == "2"
    assert parsed["b"] == "b"
    assert parsed["c"] == "3.0"
    simple = config_opts.from_parsed(parsed)
    assert simple.a == 2
    assert simple.b == "b"
    assert simple.c == 3.0

@config(variant="base")
class VariantBase:
    a: int = 1

@config(variant="v1")
class Variant1(VariantBase):
    a: int = 2
    b: str = "b1"
    c: str = "foo"

def test_variant():
    config_opts = Options.as_options(VariantBase)
    assert len(config_opts.opts) == 4
    assert set(o.name for o in config_opts.opts) == {"type", "a", "v1.b", "v1.c"}

    parsed = config_opts.parse(["--v1.c=bar", "--type=v1"])
    assert parsed["v1.c"] == "bar"
    assert parsed["type"] == "v1"

    variant1 = config_opts.from_parsed(parsed)
    assert variant1.a == 2
    assert variant1.c == "bar"
