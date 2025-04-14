from nanoconfig.utils import parse_value

import typing as ty

def test_parse_simple():
    print(parse_value("true", bool))
    print(parse_value("foo", str))
    assert True == parse_value("True", bool)
    assert "foo" == parse_value("foo", str)
    assert 60 == parse_value("60", int)

def test_parse_complex():
    assert [1,2] == parse_value("[1,2]", list)
    assert [1,2] == parse_value("[1,2]", list[int])
    assert (1,2) == parse_value("[1,2]", ty.Sequence)
    assert (1,2) == parse_value("[1,2]", ty.Sequence[int])
