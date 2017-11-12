from collections import namedtuple
from inspect import Parameter, signature


def dataclass(name, **vars):
    fields = list(vars.keys())
    defaults = tuple(vars[field] for field in fields)

    T = namedtuple(name, fields)
    T.__new__.__defaults__ = defaults
    return T


def merge_dataclass(name, items):
    vars = []
    for item in items:
        for param in signature(item.__new__).parameters.values():
            if param.name.startswith('_'):
                continue

            default = None
            if param.default != Parameter.empty:
                default = param.default

            vars.append((param.name, default))

    return dataclass(name, **dict(vars))
