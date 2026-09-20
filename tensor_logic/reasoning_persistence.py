"""Persistent derived-state vs stateless recomputation experiment support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
R = TypeVar("R")


@dataclass
class PersistenceMetrics:
    source_reads: int = 0
    derivations: int = 0
    cache_hits: int = 0
    invalidations: int = 0


class VersionedSource(Generic[K, V]):
    def __init__(self, values: dict[K, V] | None = None) -> None:
        self._values = dict(values or {})
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def read(self, key: K) -> V:
        return self._values[key]

    def set(self, key: K, value: V) -> None:
        self._values[key] = value
        self._revision += 1


class StatelessDerivedView(Generic[K, V, R]):
    def __init__(
        self,
        source: VersionedSource[K, V],
        derive: Callable[[K, V], R],
    ) -> None:
        self.source = source
        self.derive = derive
        self.metrics = PersistenceMetrics()

    def get(self, key: K) -> R:
        self.metrics.source_reads += 1
        value = self.source.read(key)
        self.metrics.derivations += 1
        return self.derive(key, value)


class MaterializedDerivedView(Generic[K, V, R]):
    def __init__(
        self,
        source: VersionedSource[K, V],
        derive: Callable[[K, V], R],
    ) -> None:
        self.source = source
        self.derive = derive
        self.metrics = PersistenceMetrics()
        self._cache: dict[K, tuple[int, R]] = {}

    def get(self, key: K) -> R:
        existing = self._cache.get(key)
        if existing is not None:
            revision, result = existing
            if revision == self.source.revision:
                self.metrics.cache_hits += 1
                return result
            self.metrics.invalidations += 1
            self._cache.pop(key, None)

        self.metrics.source_reads += 1
        value = self.source.read(key)
        self.metrics.derivations += 1
        result = self.derive(key, value)
        self._cache[key] = (self.source.revision, result)
        return result

    def invalidate(self, key: K | None = None) -> int:
        if key is None:
            count = len(self._cache)
            self._cache.clear()
        else:
            count = int(key in self._cache)
            self._cache.pop(key, None)
        self.metrics.invalidations += count
        return count
