from inspect import Parameter, signature

import attr


def dataclass(name, **vars):
    params = {}
    for var, value in vars.items():
        params[var] = attr.ib(default=value)

    return attr.make_class(name, params)


def merge_dataclass(name, items, **overrides):
    vars = []
    for item in items:
        for param in signature(item.__init__).parameters.values():
            if param.name == 'self' or param.name.startswith('_'):
                continue

            default = None
            if param.default != Parameter.empty:
                default = param.default

            vars.append((param.name, default))

    params = dict(vars)
    for key, value in overrides.items():
        params[key] = value

    return dataclass(name, **params)


# TODO: rewrite this
class HiddenFrozenView(object):
    def __init__(self, obj, mask, default_value=None):
        self.mask = mask
        self.default_value = default_value
        self.obj = obj

    def __setattr__(self, key, value):
        if 'obj' in self.__dict__:
            raise AttributeError('cannot modify frozen view')
        else:
            self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.__dict__['mask']:
            return self.__dict__['default_value']
        return getattr(self.__dict__['obj'], item)
