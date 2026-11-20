"""Logical-entity construction for the unified anomaly-detection protocol."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from specmoe.data.types import LoadedTimeSeries, TimeSeriesSegment


PROTOCOL_VERSION = "entity_60_10_30_v2"
OFFICIAL_PROTOCOL_VERSION = "official_train_val_test_v1"
PAPER_PROTOCOL_VERSION = "source_chronological_50_10_40_v1"


@dataclass(frozen=True)
class LogicalEntity:
    """One complete chronological entity assembled from loader source segments."""

    entity_id: str
    values: np.ndarray
    labels: np.ndarray
    sources: Sequence[Mapping[str, object]]
    join_offset: Optional[int]

    def __post_init__(self) -> None:
        normalized = TimeSeriesSegment(
            self.entity_id,
            self.values,
            self.labels,
        )
        try:
            source_records = tuple(self.sources)
        except TypeError as exc:
            raise ValueError("Logical entity sources must be a sequence") from exc
        if not source_records or not all(
            isinstance(record, Mapping) for record in source_records
        ):
            raise ValueError(
                "Logical entity sources must contain non-empty mappings"
            )
        normalized_sources = []
        for record in source_records:
            if set(record) != {"split", "segment_id", "length"}:
                raise ValueError(
                    "Logical entity source schema must contain split, "
                    "segment_id, and length"
                )
            split = record["split"]
            segment_id = record["segment_id"]
            length = record["length"]
            if split not in {"train", "test"}:
                raise ValueError("Logical entity source split must be train or test")
            if not isinstance(segment_id, str) or not segment_id.strip():
                raise ValueError("Logical entity source segment_id must be non-empty")
            if (
                isinstance(length, bool)
                or not isinstance(length, (int, np.integer))
                or int(length) < 1
            ):
                raise ValueError(
                    "Logical entity source length must be a positive integer"
                )
            normalized_sources.append(
                MappingProxyType(
                    {
                        "split": split,
                        "segment_id": segment_id.strip(),
                        "length": int(length),
                    }
                )
            )

        if len(normalized_sources) == 1:
            if self.join_offset is not None:
                raise ValueError(
                    "Standalone logical entities must not define join_offset"
                )
        elif len(normalized_sources) == 2:
            expected_join = int(normalized_sources[0]["length"])
            if (
                normalized_sources[0]["split"] != "train"
                or normalized_sources[1]["split"] != "test"
                or normalized_sources[0]["segment_id"]
                != normalized_sources[1]["segment_id"]
                or self.join_offset != expected_join
            ):
                raise ValueError(
                    "Paired logical entities require matching train/test "
                    "sources and join_offset equal to the train length"
                )
        else:
            raise ValueError(
                "Logical entity must contain one standalone or two paired sources"
            )

        if sum(record["length"] for record in normalized_sources) != normalized.length:
            raise ValueError(
                "Logical entity source lengths must equal the entity length"
            )
        object.__setattr__(self, "entity_id", normalized.segment_id)
        object.__setattr__(self, "values", normalized.values)
        object.__setattr__(self, "labels", normalized.labels)
        object.__setattr__(self, "sources", tuple(normalized_sources))

    @property
    def length(self) -> int:
        """Number of chronological points in the logical entity."""
        return int(self.values.shape[0])


@dataclass(frozen=True)
class ProtocolSplit:
    """Materialized train, validation, and test segments plus their manifest."""

    train_segments: Sequence[TimeSeriesSegment]
    validation_segments: Sequence[TimeSeriesSegment]
    test_segments: Sequence[TimeSeriesSegment]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("train_segments", "validation_segments", "test_segments"):
            try:
                segments = tuple(getattr(self, name))
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence") from exc
            if (
                name != "test_segments"
                and not segments
            ) or not all(
                isinstance(segment, TimeSeriesSegment) for segment in segments
            ):
                raise ValueError(
                    f"{name} must contain valid TimeSeriesSegment values"
                )
            object.__setattr__(self, name, segments)
        if not isinstance(self.manifest, Mapping):
            raise ValueError("Protocol manifest must be a mapping")
        object.__setattr__(
            self,
            "manifest",
            MappingProxyType(dict(self.manifest)),
        )


def _source_labels(segment: TimeSeriesSegment) -> np.ndarray:
    if segment.labels is None:
        return np.zeros(segment.length, dtype=np.int32)
    return segment.labels


def _source_record(split: str, segment: TimeSeriesSegment) -> dict[str, object]:
    return {
        "split": split,
        "segment_id": segment.segment_id,
        "length": segment.length,
    }


def _validated_fraction(
    value: object,
    *,
    name: str,
) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a [numerator, denominator] pair")
    numerator, denominator = value
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, (int, np.integer))
        or not isinstance(denominator, (int, np.integer))
        or int(numerator) < 1
        or int(denominator) <= int(numerator)
    ):
        raise ValueError(
            f"{name} must contain positive integers with numerator < denominator"
        )
    return int(numerator), int(denominator)


def _official_segment_metadata(
    source: TimeSeriesSegment,
    *,
    split_name: str,
    start: int,
    stop: int,
) -> dict[str, object]:
    return {
        "protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "source_split": "train" if split_name != "test" else "test",
        "source_segment_id": source.segment_id,
        "source_length": source.length,
        "split": split_name,
        "start": start,
        "stop": stop,
    }


def split_official_partitions(
    loaded: LoadedTimeSeries,
    *,
    validation_fraction: Sequence[int] = (1, 10),
    window_size: int,
) -> ProtocolSplit:
    """Split only official train sources and preserve official test sources."""
    if not isinstance(loaded, LoadedTimeSeries):
        raise TypeError("loaded must be a LoadedTimeSeries")
    numerator, denominator = _validated_fraction(
        validation_fraction,
        name="validation_fraction",
    )
    window_size = _validated_window_size(window_size)
    train_segments = []
    validation_segments = []
    test_segments = []
    train_manifest: dict[str, dict[str, object]] = {}
    test_manifest: dict[str, dict[str, object]] = {}

    for source in sorted(
        loaded.train_segments,
        key=lambda segment: segment.segment_id,
    ):
        validation_length = max(
            1,
            source.length * numerator // denominator,
        )
        train_stop = source.length - validation_length
        _validate_scored_interval(
            entity_id=source.segment_id,
            split_name="official_train",
            start=0,
            stop=train_stop,
            window_size=window_size,
            require_fixed_window=True,
        )
        _validate_scored_interval(
            entity_id=source.segment_id,
            split_name="official_validation",
            start=train_stop,
            stop=source.length,
            window_size=window_size,
            require_fixed_window=False,
        )
        train_id = f"{source.segment_id}:train"
        validation_id = f"{source.segment_id}:validation"
        train_segments.append(
            TimeSeriesSegment(
                train_id,
                source.values[:train_stop],
                metadata=_official_segment_metadata(
                    source,
                    split_name="train",
                    start=0,
                    stop=train_stop,
                ),
            )
        )
        validation_segments.append(
            TimeSeriesSegment(
                validation_id,
                source.values[train_stop:],
                metadata=_official_segment_metadata(
                    source,
                    split_name="validation",
                    start=train_stop,
                    stop=source.length,
                ),
            )
        )
        train_manifest[source.segment_id] = {
            "source": _source_record("train", source),
            "train_interval": [0, train_stop],
            "validation_interval": [train_stop, source.length],
            "train_segment_id": train_id,
            "validation_segment_id": validation_id,
        }

    for source in sorted(
        loaded.test_segments,
        key=lambda segment: segment.segment_id,
    ):
        test_id = f"{source.segment_id}:test"
        test_segments.append(
            TimeSeriesSegment(
                test_id,
                source.values,
                source.labels,
                _official_segment_metadata(
                    source,
                    split_name="test",
                    start=0,
                    stop=source.length,
                ),
            )
        )
        matching_train = train_manifest.get(source.segment_id)
        test_manifest[source.segment_id] = {
            "source": _source_record("test", source),
            "interval": [0, source.length],
            "test_segment_id": test_id,
            "validation_segment_id": (
                None
                if matching_train is None
                else matching_train["validation_segment_id"]
            ),
        }

    manifest = {
        "protocol_version": OFFICIAL_PROTOCOL_VERSION,
        "validation_fraction": [numerator, denominator],
        "training_labels_used": False,
        "train_segments": train_manifest,
        "test_segments": test_manifest,
    }
    return ProtocolSplit(
        train_segments=train_segments,
        validation_segments=validation_segments,
        test_segments=test_segments,
        manifest=manifest,
    )


def build_logical_entities(
    loaded: LoadedTimeSeries,
) -> Tuple[LogicalEntity, ...]:
    """Join matching train/test sources and preserve unmatched source entities."""
    if not isinstance(loaded, LoadedTimeSeries):
        raise TypeError("loaded must be a LoadedTimeSeries")
    train_by_id = {segment.segment_id: segment for segment in loaded.train_segments}
    test_by_id = {segment.segment_id: segment for segment in loaded.test_segments}
    entities = []
    for entity_id in sorted(set(train_by_id) | set(test_by_id)):
        train = train_by_id.get(entity_id)
        test = test_by_id.get(entity_id)
        if train is not None and test is not None:
            values = np.concatenate((train.values, test.values), axis=0)
            labels = np.concatenate(
                (_source_labels(train), _source_labels(test)),
                axis=0,
            )
            sources = (
                _source_record("train", train),
                _source_record("test", test),
            )
            join_offset = train.length
        else:
            split = "train" if train is not None else "test"
            source = train if train is not None else test
            if source is None:
                raise RuntimeError("Logical entity source lookup is inconsistent")
            values = source.values
            labels = _source_labels(source)
            sources = (_source_record(split, source),)
            join_offset = None
        entities.append(
            LogicalEntity(
                entity_id=entity_id,
                values=values,
                labels=labels,
                sources=sources,
                join_offset=join_offset,
            )
        )
    if not entities:
        raise ValueError("LoadedTimeSeries has no source segments")
    return tuple(entities)


def _validated_seed(seed: int) -> int:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("Protocol seed must be a non-negative integer")
    return int(seed)


def _validated_window_size(window_size: int) -> int:
    if (
        isinstance(window_size, (bool, np.bool_))
        or not isinstance(window_size, (int, np.integer))
        or int(window_size) < 1
    ):
        raise ValueError("Protocol window_size must be a positive integer")
    return int(window_size)


def _tail_blocks(entity_id: str, length: int) -> tuple[int, list[list[int]]]:
    train_end = length * 3 // 5
    tail_length = length - train_end
    if train_end < 1 or tail_length < 4:
        raise ValueError(
            f"Entity {entity_id!r} length {length} cannot provide a "
            "non-empty 60% training interval and four non-empty tail blocks"
        )
    base, remainder = divmod(tail_length, 4)
    blocks = []
    start = train_end
    for index in range(4):
        stop = start + base + int(index < remainder)
        blocks.append([start, stop])
        start = stop
    if start != length or any(stop <= start for start, stop in blocks):
        raise RuntimeError("Internal protocol tail-block construction failed")
    return train_end, blocks


def _validate_scored_interval(
    *,
    entity_id: str,
    split_name: str,
    start: int,
    stop: int,
    window_size: int,
    require_fixed_window: bool,
) -> None:
    length = stop - start
    if length < 1:
        raise ValueError(
            f"Entity {entity_id!r} {split_name} block must be non-empty"
        )
    if require_fixed_window and length < window_size:
        raise ValueError(
            f"Entity {entity_id!r} {split_name} block has length {length}, "
            f"but window_size={window_size} is required"
        )


def _block_segment_id(
    entity_id: str,
    *,
    split_name: str,
    block_index: Optional[int],
) -> str:
    if split_name == "train":
        if block_index is not None:
            raise ValueError("Training segment IDs cannot have a block index")
        return f"{entity_id}:train"
    if block_index is None:
        raise ValueError("Validation/test segment IDs require a block index")
    return f"{entity_id}:{split_name}:{block_index}"


def _segment_metadata(
    entity: LogicalEntity,
    *,
    split_name: str,
    start: int,
    stop: int,
    block_index: Optional[int],
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "entity_id": entity.entity_id,
        "split": split_name,
        "block_index": block_index,
        "start": start,
        "stop": stop,
        "sources": [dict(source) for source in entity.sources],
        "join_offset": entity.join_offset,
    }


def split_logical_entities(
    entities: Sequence[LogicalEntity],
    *,
    seed: int,
    window_size: int,
) -> ProtocolSplit:
    """Apply the deterministic per-entity 60%/10%/30% protocol."""
    try:
        ordered_entities = tuple(entities)
    except TypeError as exc:
        raise ValueError("entities must be a sequence") from exc
    if not ordered_entities or not all(
        isinstance(entity, LogicalEntity) for entity in ordered_entities
    ):
        raise ValueError("entities must be a non-empty LogicalEntity sequence")
    entity_ids = [entity.entity_id for entity in ordered_entities]
    if entity_ids != sorted(entity_ids) or len(entity_ids) != len(set(entity_ids)):
        raise ValueError(
            "entities must be uniquely identified in stable lexicographic order"
        )
    seed = _validated_seed(seed)
    window_size = _validated_window_size(window_size)
    rng = np.random.default_rng(seed)
    train_segments = []
    validation_segments = []
    test_segments = []
    manifest_entities: dict[str, dict[str, object]] = {}

    for entity in ordered_entities:
        train_end, tail_blocks = _tail_blocks(entity.entity_id, entity.length)
        validation_block = int(rng.integers(4))
        test_blocks = [
            block_index
            for block_index in range(4)
            if block_index != validation_block
        ]
        _validate_scored_interval(
            entity_id=entity.entity_id,
            split_name="train",
            start=0,
            stop=train_end,
            window_size=window_size,
            require_fixed_window=True,
        )
        train_segments.append(
            TimeSeriesSegment(
                _block_segment_id(
                    entity.entity_id,
                    split_name="train",
                    block_index=None,
                ),
                entity.values[:train_end],
                entity.labels[:train_end],
                _segment_metadata(
                    entity,
                    split_name="train",
                    start=0,
                    stop=train_end,
                    block_index=None,
                ),
            )
        )

        for block_index, (start, stop) in enumerate(tail_blocks):
            split_name = (
                "validation"
                if block_index == validation_block
                else "test"
            )
            _validate_scored_interval(
                entity_id=entity.entity_id,
                split_name=f"{split_name}:{block_index}",
                start=start,
                stop=stop,
                window_size=window_size,
                require_fixed_window=False,
            )
            segment = TimeSeriesSegment(
                _block_segment_id(
                    entity.entity_id,
                    split_name=split_name,
                    block_index=block_index,
                ),
                entity.values[start:stop],
                entity.labels[start:stop],
                _segment_metadata(
                    entity,
                    split_name=split_name,
                    start=start,
                    stop=stop,
                    block_index=block_index,
                ),
            )
            if split_name == "validation":
                validation_segments.append(segment)
            else:
                test_segments.append(segment)

        manifest_entities[entity.entity_id] = {
            "sources": [dict(source) for source in entity.sources],
            "join_offset": entity.join_offset,
            "length": entity.length,
            "train": [0, train_end],
            "tail_blocks": tail_blocks,
            "validation_block": validation_block,
            "test_blocks": test_blocks,
            "validation_segment_id": _block_segment_id(
                entity.entity_id,
                split_name="validation",
                block_index=validation_block,
            ),
            "test_segment_ids": [
                _block_segment_id(
                    entity.entity_id,
                    split_name="test",
                    block_index=block_index,
                )
                for block_index in test_blocks
            ],
        }

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "train_fraction": [3, 5],
        "tail_block_count": 4,
        "entities": manifest_entities,
    }
    return ProtocolSplit(
        train_segments=train_segments,
        validation_segments=validation_segments,
        test_segments=test_segments,
        manifest=manifest,
    )


def _validated_block_index(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) not in range(4)
    ):
        raise ValueError(f"{name} must be an integer in [0, 4)")
    return int(value)


def _validated_interval(value: object, *, name: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a [start, stop] interval")
    start, stop = value
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, (int, np.integer))
        or not isinstance(stop, (int, np.integer))
        or int(start) < 0
        or int(stop) <= int(start)
    ):
        raise ValueError(f"{name} must contain increasing non-negative integers")
    return [int(start), int(stop)]


def _manifest_entity_record(
    entity: LogicalEntity,
    record: object,
    *,
    window_size: int,
) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Manifest entity {entity.entity_id!r} must be a mapping")
    required = {
        "sources",
        "join_offset",
        "length",
        "train",
        "tail_blocks",
        "validation_block",
        "test_blocks",
        "validation_segment_id",
        "test_segment_ids",
    }
    if set(record) != required:
        missing = sorted(required - set(record))
        extra = sorted(set(record) - required)
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} schema mismatch: "
            f"missing={missing}, extra={extra}"
        )
    expected_sources = [dict(source) for source in entity.sources]
    if record["sources"] != expected_sources:
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} sources do not match loader"
        )
    if record["join_offset"] != entity.join_offset:
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} join_offset does not match loader"
        )
    if (
        isinstance(record["length"], bool)
        or not isinstance(record["length"], (int, np.integer))
        or int(record["length"]) != entity.length
    ):
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} length does not match loader"
        )
    expected_train_end, expected_tail_blocks = _tail_blocks(
        entity.entity_id,
        entity.length,
    )
    train_interval = _validated_interval(
        record["train"],
        name=f"Manifest entity {entity.entity_id!r} train",
    )
    if train_interval != [0, expected_train_end]:
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} train interval is invalid"
        )
    raw_tail_blocks = record["tail_blocks"]
    if (
        not isinstance(raw_tail_blocks, Sequence)
        or isinstance(raw_tail_blocks, (str, bytes))
        or len(raw_tail_blocks) != 4
    ):
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} must contain four tail blocks"
        )
    tail_blocks = [
        _validated_interval(
            block,
            name=f"Manifest entity {entity.entity_id!r} tail block {index}",
        )
        for index, block in enumerate(raw_tail_blocks)
    ]
    if tail_blocks != expected_tail_blocks:
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} tail blocks are invalid"
        )
    validation_block = _validated_block_index(
        record["validation_block"],
        name=f"Manifest entity {entity.entity_id!r} validation_block",
    )
    raw_test_blocks = record["test_blocks"]
    if (
        not isinstance(raw_test_blocks, Sequence)
        or isinstance(raw_test_blocks, (str, bytes))
        or len(raw_test_blocks) != 3
    ):
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} test_blocks must contain three indexes"
        )
    test_blocks = [
        _validated_block_index(
            block,
            name=f"Manifest entity {entity.entity_id!r} test block index",
        )
        for block in raw_test_blocks
    ]
    if test_blocks != sorted(test_blocks) or set(test_blocks) != (
        set(range(4)) - {validation_block}
    ):
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} test blocks are invalid"
        )
    expected_validation_id = _block_segment_id(
        entity.entity_id,
        split_name="validation",
        block_index=validation_block,
    )
    if record["validation_segment_id"] != expected_validation_id:
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} validation segment ID is invalid"
        )
    expected_test_ids = [
        _block_segment_id(
            entity.entity_id,
            split_name="test",
            block_index=block_index,
        )
        for block_index in test_blocks
    ]
    if record["test_segment_ids"] != expected_test_ids:
        raise ValueError(
            f"Manifest entity {entity.entity_id!r} test segment IDs are invalid"
        )
    _validate_scored_interval(
        entity_id=entity.entity_id,
        split_name="train",
        start=0,
        stop=expected_train_end,
        window_size=window_size,
        require_fixed_window=True,
    )
    for block_index, (start, stop) in enumerate(tail_blocks):
        _validate_scored_interval(
            entity_id=entity.entity_id,
            split_name=(
                "validation"
                if block_index == validation_block
                else "test"
            )
            + f":{block_index}",
            start=start,
            stop=stop,
            window_size=window_size,
            require_fixed_window=False,
        )
    return {
        "sources": expected_sources,
        "join_offset": entity.join_offset,
        "length": entity.length,
        "train": train_interval,
        "tail_blocks": tail_blocks,
        "validation_block": validation_block,
        "test_blocks": test_blocks,
        "validation_segment_id": expected_validation_id,
        "test_segment_ids": expected_test_ids,
    }


def _validated_manifest_records(
    loaded: LoadedTimeSeries,
    manifest: Mapping[str, object],
    *,
    window_size: int,
) -> tuple[tuple[LogicalEntity, ...], dict[str, dict[str, object]]]:
    if not isinstance(manifest, Mapping):
        raise ValueError("Protocol manifest must be a mapping")
    required = {
        "protocol_version",
        "seed",
        "train_fraction",
        "tail_block_count",
        "entities",
    }
    if set(manifest) != required:
        missing = sorted(required - set(manifest))
        extra = sorted(set(manifest) - required)
        raise ValueError(
            "Protocol manifest schema mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Protocol manifest version is unsupported")
    _validated_seed(manifest["seed"])
    if manifest["train_fraction"] != [3, 5]:
        raise ValueError("Protocol manifest train_fraction must be [3, 5]")
    if manifest["tail_block_count"] != 4:
        raise ValueError("Protocol manifest tail_block_count must be 4")
    raw_entities = manifest["entities"]
    if not isinstance(raw_entities, Mapping):
        raise ValueError("Protocol manifest entities must be a mapping")
    entities = build_logical_entities(loaded)
    expected_ids = [entity.entity_id for entity in entities]
    if list(raw_entities) != expected_ids:
        raise ValueError(
            "Protocol manifest entity IDs/order do not match the loader"
        )
    records = {
        entity.entity_id: _manifest_entity_record(
            entity,
            raw_entities[entity.entity_id],
            window_size=window_size,
        )
        for entity in entities
    }
    return entities, records


def _validated_official_manifest_records(
    loaded: LoadedTimeSeries,
    manifest: Mapping[str, object],
    *,
    window_size: int,
) -> tuple[
    dict[str, tuple[TimeSeriesSegment, dict[str, object]]],
    dict[str, tuple[TimeSeriesSegment, dict[str, object]]],
]:
    required = {
        "protocol_version",
        "validation_fraction",
        "training_labels_used",
        "train_segments",
        "test_segments",
    }
    if set(manifest) != required:
        missing = sorted(required - set(manifest))
        extra = sorted(set(manifest) - required)
        raise ValueError(
            "Official protocol manifest schema mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if manifest["protocol_version"] != OFFICIAL_PROTOCOL_VERSION:
        raise ValueError("Official protocol manifest version is unsupported")
    numerator, denominator = _validated_fraction(
        manifest["validation_fraction"],
        name="Official protocol validation_fraction",
    )
    if manifest["training_labels_used"] is not False:
        raise ValueError(
            "Official protocol training_labels_used must be false"
        )
    raw_train = manifest["train_segments"]
    raw_test = manifest["test_segments"]
    if not isinstance(raw_train, Mapping) or not isinstance(raw_test, Mapping):
        raise ValueError(
            "Official protocol train/test segment records must be mappings"
        )

    train_sources = {
        segment.segment_id: segment
        for segment in sorted(
            loaded.train_segments,
            key=lambda item: item.segment_id,
        )
    }
    test_sources = {
        segment.segment_id: segment
        for segment in sorted(
            loaded.test_segments,
            key=lambda item: item.segment_id,
        )
    }
    if list(raw_train) != list(train_sources):
        raise ValueError(
            "Official protocol train source IDs/order do not match the loader"
        )
    if list(raw_test) != list(test_sources):
        raise ValueError(
            "Official protocol test source IDs/order do not match the loader"
        )

    validated_train = {}
    for segment_id, source in train_sources.items():
        record = raw_train[segment_id]
        expected_keys = {
            "source",
            "train_interval",
            "validation_interval",
            "train_segment_id",
            "validation_segment_id",
        }
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise ValueError(
                f"Official train source {segment_id!r} schema mismatch"
            )
        if record["source"] != _source_record("train", source):
            raise ValueError(
                f"Official train source {segment_id!r} identity/length mismatch"
            )
        validation_length = max(
            1,
            source.length * numerator // denominator,
        )
        train_stop = source.length - validation_length
        expected_train = [0, train_stop]
        expected_validation = [train_stop, source.length]
        if _validated_interval(
            record["train_interval"],
            name=f"Official train source {segment_id!r} train_interval",
        ) != expected_train:
            raise ValueError(
                f"Official train source {segment_id!r} train interval mismatch"
            )
        if _validated_interval(
            record["validation_interval"],
            name=f"Official train source {segment_id!r} validation_interval",
        ) != expected_validation:
            raise ValueError(
                f"Official train source {segment_id!r} validation interval mismatch"
            )
        expected_train_id = f"{segment_id}:train"
        expected_validation_id = f"{segment_id}:validation"
        if record["train_segment_id"] != expected_train_id:
            raise ValueError(
                f"Official train source {segment_id!r} train segment ID mismatch"
            )
        if record["validation_segment_id"] != expected_validation_id:
            raise ValueError(
                f"Official train source {segment_id!r} validation segment ID mismatch"
            )
        _validate_scored_interval(
            entity_id=segment_id,
            split_name="official_train",
            start=0,
            stop=train_stop,
            window_size=window_size,
            require_fixed_window=True,
        )
        _validate_scored_interval(
            entity_id=segment_id,
            split_name="official_validation",
            start=train_stop,
            stop=source.length,
            window_size=window_size,
            require_fixed_window=False,
        )
        validated_train[segment_id] = (
            source,
            {
                "train_interval": expected_train,
                "validation_interval": expected_validation,
                "train_segment_id": expected_train_id,
                "validation_segment_id": expected_validation_id,
            },
        )

    validated_test = {}
    for segment_id, source in test_sources.items():
        record = raw_test[segment_id]
        expected_keys = {
            "source",
            "interval",
            "test_segment_id",
            "validation_segment_id",
        }
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise ValueError(
                f"Official test source {segment_id!r} schema mismatch"
            )
        if record["source"] != _source_record("test", source):
            raise ValueError(
                f"Official test source {segment_id!r} identity/length mismatch"
            )
        expected_interval = [0, source.length]
        if _validated_interval(
            record["interval"],
            name=f"Official test source {segment_id!r} interval",
        ) != expected_interval:
            raise ValueError(
                f"Official test source {segment_id!r} interval mismatch"
            )
        expected_test_id = f"{segment_id}:test"
        if record["test_segment_id"] != expected_test_id:
            raise ValueError(
                f"Official test source {segment_id!r} segment ID mismatch"
            )
        matching_train = validated_train.get(segment_id)
        expected_validation_id = (
            None
            if matching_train is None
            else matching_train[1]["validation_segment_id"]
        )
        if record["validation_segment_id"] != expected_validation_id:
            raise ValueError(
                f"Official test source {segment_id!r} validation reference mismatch"
            )
        _validate_scored_interval(
            entity_id=segment_id,
            split_name="official_test",
            start=0,
            stop=source.length,
            window_size=window_size,
            require_fixed_window=False,
        )
        validated_test[segment_id] = (
            source,
            {
                "interval": expected_interval,
                "test_segment_id": expected_test_id,
                "validation_segment_id": expected_validation_id,
            },
        )
    return validated_train, validated_test


def _materialize_official_manifest_segments(
    loaded: LoadedTimeSeries,
    manifest: Mapping[str, object],
    *,
    window_size: int,
    split_name: str,
) -> Tuple[TimeSeriesSegment, ...]:
    validated_train, validated_test = _validated_official_manifest_records(
        loaded,
        manifest,
        window_size=window_size,
    )
    if split_name == "validation":
        result = []
        for source, record in validated_train.values():
            start, stop = record["validation_interval"]
            result.append(
                TimeSeriesSegment(
                    record["validation_segment_id"],
                    source.values[start:stop],
                    metadata=_official_segment_metadata(
                        source,
                        split_name="validation",
                        start=start,
                        stop=stop,
                    ),
                )
            )
        return tuple(result)
    if split_name == "test":
        result = []
        for source, record in validated_test.values():
            start, stop = record["interval"]
            result.append(
                TimeSeriesSegment(
                    record["test_segment_id"],
                    source.values[start:stop],
                    (
                        None
                        if source.labels is None
                        else source.labels[start:stop]
                    ),
                    _official_segment_metadata(
                        source,
                        split_name="test",
                        start=start,
                        stop=stop,
                    ),
                )
            )
        return tuple(result)
    raise ValueError("Manifest materialization split must be validation or test")


def _materialize_manifest_segments(
    loaded: LoadedTimeSeries,
    manifest: Mapping[str, object],
    *,
    window_size: int,
    split_name: str,
) -> Tuple[TimeSeriesSegment, ...]:
    if split_name not in {"validation", "test"}:
        raise ValueError("Manifest materialization split must be validation or test")
    window_size = _validated_window_size(window_size)
    if manifest.get("protocol_version") == PAPER_PROTOCOL_VERSION:
        expected = validate_paper_manifest(manifest, window_size=window_size)
        rebuilt = split_paper_partitions(loaded, window_size=window_size)
        if dict(rebuilt.manifest) != expected:
            raise ValueError("Loaded sources differ from the recorded paper protocol")
        return getattr(rebuilt, f"{split_name}_segments")
    if manifest.get("protocol_version") == OFFICIAL_PROTOCOL_VERSION:
        return _materialize_official_manifest_segments(
            loaded,
            manifest,
            window_size=window_size,
            split_name=split_name,
        )
    entities, records = _validated_manifest_records(
        loaded,
        manifest,
        window_size=window_size,
    )
    result = []
    for entity in entities:
        record = records[entity.entity_id]
        block_indices = (
            [record["validation_block"]]
            if split_name == "validation"
            else record["test_blocks"]
        )
        for block_index in block_indices:
            start, stop = record["tail_blocks"][block_index]
            result.append(
                TimeSeriesSegment(
                    _block_segment_id(
                        entity.entity_id,
                        split_name=split_name,
                        block_index=block_index,
                    ),
                    entity.values[start:stop],
                    entity.labels[start:stop],
                    _segment_metadata(
                        entity,
                        split_name=split_name,
                        start=start,
                        stop=stop,
                        block_index=block_index,
                    ),
                )
            )
    return tuple(result)


def materialize_manifest_validation_segments(
    loaded: LoadedTimeSeries,
    manifest: Mapping[str, object],
    *,
    window_size: int,
) -> Tuple[TimeSeriesSegment, ...]:
    """Rebuild the manifest-designated independent validation blocks."""
    return _materialize_manifest_segments(
        loaded,
        manifest,
        window_size=window_size,
        split_name="validation",
    )


def materialize_manifest_test_segments(
    loaded: LoadedTimeSeries,
    manifest: Mapping[str, object],
    *,
    window_size: int,
) -> Tuple[TimeSeriesSegment, ...]:
    """Rebuild the manifest-designated independent test blocks."""
    return _materialize_manifest_segments(
        loaded,
        manifest,
        window_size=window_size,
        split_name="test",
    )


def _paper_manifest_from_sources(sources, *, window_size):
    """Describe chronological partitions without concatenating source sequences."""
    window_size = _validated_window_size(window_size)
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Paper protocol sources must be a non-empty dictionary")
    records = {}
    for source_id in sorted(sources):
        source = sources[source_id]
        if not isinstance(source, dict) or set(source) != {"split", "segment_id", "length"}:
            raise ValueError("Paper protocol source schema is invalid")
        if source["split"] not in {"train", "test"}:
            raise ValueError("Paper protocol source split must be train or test")
        if not isinstance(source["segment_id"], str) or not source["segment_id"].strip():
            raise ValueError("Paper protocol source segment ID must be non-empty")
        if source_id != f"{source['split']}:{source['segment_id']}":
            raise ValueError("Paper protocol source identity is invalid")
        n = source["length"]
        if type(n) is not int or n < 1:
            raise ValueError("Paper protocol source length must be a positive integer")
        train_stop = n // 2
        validation_stop = train_stop + n // 10
        if train_stop < window_size or n // 10 < 4 or n - validation_stop < 4:
            raise ValueError(
                f"Source {source_id!r} length {n} cannot support 50/10/40 "
                f"with a training window of {window_size} and at least four "
                "time points in each evaluation partition; no padding or borrowing is allowed"
            )
        records[source_id] = {
            "source": dict(source),
            "train_interval": [0, train_stop],
            "validation_interval": [train_stop, validation_stop],
            "test_interval": [validation_stop, n],
            "train_segment_id": f"{source_id}:train",
            "validation_segment_id": f"{source_id}:validation",
            "test_segment_id": f"{source_id}:test",
        }
    return {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "allocation": [5, 1, 4],
        "allocation_unit": "independent_source_sequence",
        "rounding": "floor_train_and_validation_remainder_test",
        "training_labels_used": True,
        "sources": records,
    }


def validate_paper_manifest(manifest, *, window_size):
    """Reject changed boundaries, identities, ratios and malformed checkpoint metadata."""
    import json

    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("sources"), dict):
        raise ValueError("Paper protocol manifest must contain source records")
    sources = {}
    for source_id, record in manifest["sources"].items():
        if not isinstance(source_id, str) or not isinstance(record, dict):
            raise ValueError("Paper protocol source record is invalid")
        sources[source_id] = record.get("source")
    expected = _paper_manifest_from_sources(sources, window_size=window_size)
    # JSON comparison distinguishes booleans from integers as well as extra fields.
    if json.dumps(dict(manifest), sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Paper protocol manifest does not match deterministic 50/10/40 boundaries")
    return expected


def split_paper_partitions(loaded: LoadedTimeSeries, *, window_size: int) -> ProtocolSplit:
    """Split each independent source 50/10/40 before normal-window filtering.

    Official train/test names identify input files only. Both are repartitioned;
    no window spans an original source, entity, excluded gap or new split.
    Labels are retained so training can reject every window containing anomalies.
    """
    if not isinstance(loaded, LoadedTimeSeries):
        raise TypeError("loaded must be a LoadedTimeSeries")
    source_segments = {
        f"{source_split}:{segment.segment_id}": (source_split, segment)
        for source_split, segments in (
            ("train", loaded.train_segments), ("test", loaded.test_segments)
        )
        for segment in segments
    }
    manifest = _paper_manifest_from_sources(
        {key: _source_record(source_split, segment)
         for key, (source_split, segment) in source_segments.items()},
        window_size=window_size,
    )
    partitions = {name: [] for name in ("train", "validation", "test")}
    for source_id, record in manifest["sources"].items():
        source_split, segment = source_segments[source_id]
        if source_split == "test" and segment.labels is None:
            raise ValueError(f"Test source {source_id!r} requires labels for normal-only training")
        labels = _source_labels(segment)
        for split_name, segments in partitions.items():
            start, stop = record[f"{split_name}_interval"]
            segments.append(TimeSeriesSegment(
                record[f"{split_name}_segment_id"],
                segment.values[start:stop], labels[start:stop],
                {"protocol_version": PAPER_PROTOCOL_VERSION,
                 "source_split": source_split,
                 "source_segment_id": segment.segment_id,
                 "source_length": segment.length,
                 "split": split_name, "start": start, "stop": stop},
            ))
    return ProtocolSplit(
        train_segments=partitions["train"],
        validation_segments=partitions["validation"],
        test_segments=partitions["test"], manifest=manifest,
    )
