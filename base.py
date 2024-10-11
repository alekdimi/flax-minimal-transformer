from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T")


class MakeableConfig[T](ABC):
    @abstractmethod
    def make(self) -> T: ...
