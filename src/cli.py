import argparse
import inspect
import re
import sys
from dataclasses import _MISSING_TYPE, fields
from typing import Any, Generic, TypeVar, TypeVarTuple, cast, get_args, get_origin

import yaml

from .serialize import SerializableData, serialize

TYArg = TypeVar("TYArg")


def get_attribute_docs(t: type) -> dict[str, str]:
    docs = {}
    try:
        lines = inspect.getsource(t).splitlines()
    except:
        return docs
    last_attr = ""
    for line in lines:
        line = line.strip()
        try:
            s = eval(line)
            docs[last_attr] = s
        except SyntaxError:
            last_attr = line.split(":")[0]
    return docs


class YArg(Generic[TYArg]):
    def __init__(self, t: type[TYArg]) -> None:
        self._type = t

    def __call__(self, s: str) -> TYArg:
        origin = get_origin(self._type) or self._type

        if issubclass(origin, SerializableData):
            try:
                return origin(**yaml.safe_load(s))
            except Exception as e:
                raise argparse.ArgumentTypeError(str(e))

        if origin == str:
            return cast(TYArg, s)

        try:
            return origin(eval(s))
        except Exception as e:
            raise argparse.ArgumentTypeError(str(e))

    def __repr__(self) -> str:
        return str(self._type)


re_split_args = re.compile(r"\s+(?=--[^\s])")
re_split_arg_value = re.compile(r"(--[^\s]+)\s*(.+)\s*")

TData = TypeVarTuple("TData")


class Cli(Generic[*TData]):
    def __init__(self, data: type[tuple[*TData]]) -> None:
        super().__init__()
        self._data = tuple(get_args(data))
        self._parser = argparse.ArgumentParser(
            argument_default=argparse.SUPPRESS,
            usage="%(prog)s [--param_name value]*",
            exit_on_error=True,
            formatter_class=argparse.HelpFormatter,
        )
        for d in self._data:
            self.add_arguments(d)

    @staticmethod
    def format_type(t: type) -> str:
        if t == Ellipsis:
            return "..."

        origin = get_origin(t) or t
        args = get_args(t)
        type_name = origin.__name__
        if len(args) > 0:
            sargs = ", ".join([Cli.format_type(arg) for arg in args])
            type_name += f"[{sargs}]"
        return type_name

    def add_arguments(self, data: type[SerializableData]) -> None:
        group = self._parser.add_argument_group(data.__name__)
        docs = get_attribute_docs(data)
        for f in fields(data):
            default = f.default
            required = False

            if isinstance(default, _MISSING_TYPE):
                default = f.default_factory
                if not isinstance(default, _MISSING_TYPE):
                    default = serialize(f.type, default())

            if isinstance(default, _MISSING_TYPE):
                default = ""
                required = True

            if f.name in docs:
                doc = ": " + docs[f.name]
            else:
                doc = ""

            group.add_argument(
                f"--{f.name}",
                type=YArg(f.type),
                metavar=self.format_type(f.type),
                help=f"{default}{doc}",
                required=required,
            )

    def parse_args(self, argv: list[str] | None = None) -> dict[str, Any]:
        if argv == None:
            argv = [a.strip() for a in sys.argv[1:]]
            argv = re_split_args.split(" ".join(sys.argv[1:]))
            argv = [a.strip() for a in argv]
            argv = [a for a in argv if len(a) > 0]
            args: list[str] = []
            for arg in argv:
                split = re_split_arg_value.match(arg)
                if split is None:
                    args.append(arg)
                else:
                    args.extend((split[1], split[2]))
        else:
            args = argv
        return dict(self._parser.parse_args(args)._get_kwargs())

    def parse(self, argv: list[str] | None = None) -> tuple[*TData]:
        args = self.parse_args(argv)
        keys = [[f.name for f in fields(d)] for d in self._data]
        dicts = [{k: args[k] for k in keys[i] if k in args} for i in range(len(keys))]
        return tuple(self._data[i](**dicts[i]) for i in range(len(self._data))) # type: ignore
