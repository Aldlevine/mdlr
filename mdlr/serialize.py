from dataclasses import KW_ONLY, dataclass, fields
from types import GenericAlias
from typing import (
    Any,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    Self,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)


def is_concrete_type(
    cls: type | dict[str, type] | tuple[type, ...] | list[type]
) -> tuple[bool, list[type]]:
    bad_attrs: list[type] = []

    # ignore special types/values
    if get_origin(cls) in (ClassVar,) or cls in (KW_ONLY,):
        return False, []

    if isinstance(cls, dict):
        for t in cls.values():
            if not is_concrete_type(t):
                bad_attrs.append(t)
        return len(bad_attrs) == 0, bad_attrs

    if isinstance(cls, (tuple, list)):
        for t in cls:
            if not is_concrete_type(t):
                bad_attrs.append(t)
        return len(bad_attrs) == 0, bad_attrs

    if cls == Ellipsis:
        return True, []

    # HACK
    try:
        if issubclass(cls, StateDictSerializable):
            return False, []
    except TypeError:
        ...

    try:
        if (
            issubclass(cls, (Generic, GenericAlias))
            or cls in (dict, list, tuple)
            or get_origin(cls) in (dict, list, tuple)
        ):
            if get_origin(cls) is None:
                bad_attrs.append(cls)

            for arg in get_args(cls):
                ic, ba = is_concrete_type(arg)
                if not ic:
                    bad_attrs.extend(ba)

    except TypeError:
        bad_attrs.append(cls)
    if hasattr(cls, "__annotations__"):
        types = get_type_hints(cls)
        for k, annot in types.items():
            ic, ba = is_concrete_type(annot)
            if not ic:
                bad_attrs.extend(ba)

    return len(bad_attrs) == 0, bad_attrs

def get_or_none(obj: Any | None, k: str) -> Any:
    if not isinstance(obj, dict):
        return None
    return obj.get(k, None)

def getattr_or_none(obj: Any | None, k: str) -> Any:
    if obj == None:
        return None
    return getattr(obj, k)


def getidx_or_none(obj: Any | None, k: int) -> Any:
    if obj == None:
        return None
    if len(obj) <= k:
        return None
    return obj[k]


def assert_err(cond: bool, err_type: type[BaseException], *args, **kwargs) -> None:
    if not cond:
        raise err_type(*args, **kwargs)


@dataclass
class SerializableData:
    def __init_subclass__(cls) -> None:
        valid, invalid_types = is_concrete_type(cls)
        ivstr = [
            str(t.__qualname__ if hasattr(t, "__qualname__") else t)
            for t in invalid_types
        ]
        assert_err(
            valid, TypeError, f"All types must be concrete: [{', '.join(ivstr)}]"
        )

        attrs = [
            k for k in cls.__dict__ if not k.startswith("__") and not k.endswith("__")
        ]

        no_annotations_msg = "All fields must be annotated to support (de)serialization"
        types = get_type_hints(cls)
        assert_err(len(types) >= len(attrs), TypeError, no_annotations_msg)
        for a in attrs:
            assert_err(a in types, TypeError, no_annotations_msg)

    @classmethod
    def serialize(cls, self: Self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            k = f.name
            types = get_type_hints(cls)
            assert_err(k in types, ValueError, f"{cls} contains untyped data {k}")
            out[k] = serialize(types[k], getattr(self, k))
        return out

    @classmethod
    def deserialize(cls, data: dict[str, Any], state: Optional[Self] = None) -> Self:
        out: dict[str, Any] = {}
        for k, v in data.items():
            types = get_type_hints(cls)
            assert_err(k in types, ValueError, f"{cls} contains untyped data {k}")
            out[k] = deserialize(types[k], v, getattr_or_none(state, k))
        return cls(**out)


@runtime_checkable
class StateDictSerializable(Protocol):
    def state_dict(self, **kwargs) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any], *args, **kwargs) -> Any:
        ...


Simple_t = bool | int | float | complex | str
DiscreteSerializable_t = Simple_t | SerializableData
Serializable_t = (
    DiscreteSerializable_t
    | StateDictSerializable
    | list["Serializable_t"]
    | tuple["Serializable_t"]
    | dict[str, "Serializable_t"]
)

Serialized_t = (
    Simple_t | list["Serialized_t"] | tuple["Serialized_t"] | dict[str, "Serialized_t"]
)


def serialize(srl_type: type, srl: Serializable_t) -> Serialized_t:
    # HACK
    try:
        if issubclass(srl_type, StateDictSerializable):
            assert isinstance(srl, StateDictSerializable)
            return srl.state_dict()
    except TypeError:
        ...

    if issubclass(srl_type, SerializableData):
        assert isinstance(srl, SerializableData)
        return srl_type.serialize(srl)

    origin = get_origin(srl_type)
    args = get_args(srl_type)

    if origin == dict:
        assert isinstance(srl, dict)
        if isinstance(args[0], dict):
            # TypedDict
            return {k: serialize(args[0][k], v) for k, v in srl.items()}
        else:
            return {k: serialize(args[1], v) for k, v in srl.items()}

    if origin == list:
        assert isinstance(srl, list)
        return [serialize(args[0], v) for v in srl]

    if origin == tuple or issubclass(srl_type, tuple):  # because of named tuples
        assert isinstance(srl, tuple)
        if args is not None and len(args) == 2 and args[1] == ...:
            return tuple(serialize(args[0], v) for v in srl)
        assert_err(
            len(args) == len(srl), TypeError, "Non variadic tuple is incorrect length"
        )
        return tuple(serialize(args[i], v) for i, v in enumerate(srl))

    assert not issubclass(srl_type, StateDictSerializable)
    return srl_type(srl)


def deserialize(
    srl_type: type, data: Serialized_t, state: Optional[Serializable_t] = None
) -> Serializable_t:
    if issubclass(srl_type, SerializableData):
        assert isinstance(data, dict)
        assert isinstance(state, SerializableData) or state is None
        return srl_type.deserialize(data, state)

    # HACK
    try:
        if issubclass(srl_type, StateDictSerializable):
            assert isinstance(data, dict)
            assert isinstance(
                state, StateDictSerializable
            ), "TorchSerializable state is required to deserialize Torch object"
            state.load_state_dict(data)
            return state
    except TypeError:
        ...

    origin = get_origin(srl_type)
    args = get_args(srl_type)

    if origin == dict:
        assert isinstance(data, dict)

        if isinstance(args[0], dict):
            # TypedDict
            return {
                # k: deserialize(args[0][k], v, getattr_or_none(state, k))
                k: deserialize(args[0][k], v, get_or_none(state, k))
                for k, v in data.items()
            }
        else:
            return {
                # k: deserialize(args[1], v, getattr_or_none(state, k))
                k: deserialize(args[1], v, get_or_none(state, k))
                for k, v in data.items()
            }

    if origin == list:
        assert isinstance(data, list)
        return [
            deserialize(args[0], v, getidx_or_none(state, i))
            for i, v in enumerate(data)
        ]

    if origin == tuple or issubclass(srl_type, tuple):
        assert isinstance(
            data, (tuple, list)
        )  # we include list too because json.dump will greate an array/list for a tuple
        if args is not None and len(args) == 2 and args[1] == ...:
            return tuple(
                deserialize(args[0], v, getidx_or_none(state, i))
                for i, v in enumerate(data)
            )
        assert_err(
            len(args) == len(data), TypeError, "Non variadic tuple is incorrect length"
        )
        return tuple(
            deserialize(args[i], v, getidx_or_none(state, i))
            for i, v in enumerate(data)
        )

    return data
