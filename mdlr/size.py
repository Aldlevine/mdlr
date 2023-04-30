from typing import TypeVar, TypeVarTuple, Self, overload


Size1_t = tuple[int]
Size2_t = tuple[int, int]
Size3_t = tuple[int, int, int]
Size4_t = tuple[int, int, int, int]
SizeN_t = tuple[int, ...]
Size1_p = Size1_t | int
Size2_p = Size2_t | int
Size3_p = Size3_t | int
Size4_p = Size4_t | int
SizeN_p = SizeN_t | int

# T = TypeVar("T", Size1_t, Size2_t, Size3_t, Size4_t, SizeN_t)
# P = TypeVar("P", Size1_p, Size2_p, Size3_p, Size4_p, SizeN_p)

T = TypeVarTuple("T")
class Size(tuple[*T]):
    def __add__(self, other: Self) -> Self:
        return Size[*T](a + b for a,b in zip(self, other)) # type: ignore
    def __sub__(self, other: Self) -> Self:
        return Size[*T](a - b for a,b in zip(self, other)) # type: ignore
    
def size1(i: Size1_p) -> Size1_t:
    ...

def size2(i: Size2_p) -> Size2_t:
    ...

def size3(i: Size3_p) -> Size3_t:
    ...

def size4(i: Size4_p) -> Size4_t:
    ...


sz = Size((1, 2, 3))
print(sz + sz)

# def size(s: P) -> T:
# def size(*args: *T) -> Size[*T]:
#     return Size[*T](args)
