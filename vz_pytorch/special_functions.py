from typing import Tuple, Any


__all__ = ["FUNCTIONS"]


def _slice_notation(s):
    return f'{s.start if s.start is not None else ""}:{s.stop if s.stop is not None else ""}'


def _binary_op(op, flip=False):
    def _fn(args, kwargs):
        return (args[0], args[1]), op

    return _fn


def _getitem(args, kwargs) -> Tuple[Tuple[Any], str]:
    return (args[0],), f"self[{', '.join([_slice_notation(s) if isinstance(s, slice) else str(s) for s in args[1]])}]"


FUNCTIONS = {
    "__add__": _binary_op('+'),
    "__sub__": _binary_op('-'),
    "__mul__": _binary_op('*'),
    "__div__": _binary_op('/'),
    "__floordiv__": _binary_op('//'),
    "__truediv__": _binary_op('/'),
    "__mod__": _binary_op('%'),
    "__divmod__": _binary_op('divmod'),
    "__pow__": _binary_op('**'),
    "__lshift__": _binary_op('<<'),
    "__rshift__": _binary_op('>>'),
    "__and__": _binary_op('&'),
    "__or__": _binary_op('|'),
    "__xor__": _binary_op('^'),
    "__getitem__": _getitem,
}
FUNCTIONS = {
    **FUNCTIONS,
    **{
          "__r" + fn_name.lstrip("__"): FUNCTIONS[fn_name] for fn_name in FUNCTIONS
      },
    **{
          "__i" + fn_name.lstrip("__"): FUNCTIONS[fn_name] for fn_name in FUNCTIONS
      },
}
