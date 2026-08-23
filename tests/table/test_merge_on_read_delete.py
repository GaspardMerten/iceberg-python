# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import warnings
from pathlib import PosixPath

import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.manifest import POSITIONAL_DELETE_SCHEMA, DataFile, DataFileContent, ManifestContent
from pyiceberg.table import Table, TableProperties
from pyiceberg.table.delete_file_index import PATH_FIELD_ID
from pyiceberg.table.snapshots import ADDED_POSITION_DELETE_FILES, ADDED_POSITION_DELETES, Operation
from pyiceberg.types import LongType
from tests.catalog.test_base import InMemoryCatalog

SCHEMA = pa.schema([("id", pa.int32()), ("name", pa.string())])


@pytest.fixture
def catalog(tmp_path: PosixPath) -> InMemoryCatalog:
    catalog = InMemoryCatalog("test.in_memory.catalog", warehouse=tmp_path.absolute().as_posix())
    catalog.create_namespace("default")
    return catalog


def _create_table(catalog: Catalog, identifier: str, format_version: int = 2, merge_on_read: bool = True) -> Table:
    properties = {"format-version": str(format_version)}
    if merge_on_read:
        properties[TableProperties.DELETE_MODE] = TableProperties.DELETE_MODE_MERGE_ON_READ

    table = catalog.create_table(identifier, SCHEMA, properties=properties)
    table.append(
        pa.Table.from_pylist(
            [{"id": row, "name": f"name-{row}"} for row in range(1, 6)],
            schema=SCHEMA,
        )
    )
    return table


def _data_files(table: Table, content: DataFileContent) -> list[DataFile]:
    snapshot = table.metadata.current_snapshot()
    assert snapshot is not None
    return [
        entry.data_file
        for manifest in snapshot.manifests(table.io)
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True)
        if entry.data_file.content == content
    ]


def test_position_delete_schema_types_pos_as_long() -> None:
    """The spec types `pos` as a long, an int overflows on files with more than 2^31 rows."""
    assert POSITIONAL_DELETE_SCHEMA.find_field("pos").field_type == LongType()


def test_delete_writes_position_deletes_instead_of_rewriting(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_merge_on_read_delete")
    (data_file,) = _data_files(table, DataFileContent.DATA)

    table.delete("id = 3")

    # The data file is untouched, the delete is recorded next to it
    assert [file.file_path for file in _data_files(table, DataFileContent.DATA)] == [data_file.file_path]
    delete_files = _data_files(table, DataFileContent.POSITION_DELETES)
    assert len(delete_files) == 1
    assert delete_files[0].record_count == 1

    assert table.scan().to_arrow() == pa.Table.from_pylist(
        [{"id": row, "name": f"name-{row}"} for row in [1, 2, 4, 5]],
        schema=SCHEMA,
    )


def test_delete_does_not_warn_about_falling_back(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_merge_on_read_no_warning")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table.delete("id = 3")


def test_delete_writes_a_deletes_manifest(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_merge_on_read_manifest")

    table.delete("id = 3")

    snapshot = table.metadata.current_snapshot()
    assert snapshot is not None
    assert snapshot.summary is not None
    assert snapshot.summary.operation == Operation.OVERWRITE
    assert snapshot.summary[ADDED_POSITION_DELETES] == "1"
    assert snapshot.summary[ADDED_POSITION_DELETE_FILES] == "1"

    contents = [manifest.content for manifest in snapshot.manifests(table.io)]
    assert ManifestContent.DELETES in contents
    assert ManifestContent.DATA in contents


def test_position_delete_file_has_exact_path_bounds(catalog: Catalog) -> None:
    """The read side only pins a delete to a single data file when its `file_path` bounds are exact."""
    table = _create_table(catalog, "default.test_merge_on_read_bounds")
    (data_file,) = _data_files(table, DataFileContent.DATA)

    table.delete("id = 3")

    (delete_file,) = _data_files(table, DataFileContent.POSITION_DELETES)
    assert delete_file.lower_bounds[PATH_FIELD_ID].decode("utf-8") == data_file.file_path
    assert delete_file.upper_bounds[PATH_FIELD_ID].decode("utf-8") == data_file.file_path


def test_successive_deletes_accumulate(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_merge_on_read_successive")

    table.delete("id = 2")
    table.delete("id = 4")

    assert len(_data_files(table, DataFileContent.POSITION_DELETES)) == 2
    assert table.scan().to_arrow() == pa.Table.from_pylist(
        [{"id": row, "name": f"name-{row}"} for row in [1, 3, 5]],
        schema=SCHEMA,
    )


def test_delete_of_a_whole_file_drops_the_file(catalog: Catalog) -> None:
    """A file that matches entirely is still dropped outright, no delete file is needed."""
    table = _create_table(catalog, "default.test_merge_on_read_whole_file")

    table.delete("id >= 1")

    assert _data_files(table, DataFileContent.DATA) == []
    assert _data_files(table, DataFileContent.POSITION_DELETES) == []
    assert len(table.scan().to_arrow()) == 0


def test_delete_falls_back_to_copy_on_write_on_v1(catalog: Catalog) -> None:
    """A v1 table cannot store delete manifests, so it keeps rewriting the data files."""
    table = _create_table(catalog, "default.test_merge_on_read_v1", format_version=1)

    with pytest.warns(UserWarning, match="falling back to copy-on-write"):
        table.delete("id = 3")

    assert _data_files(table, DataFileContent.POSITION_DELETES) == []
    assert table.scan().to_arrow() == pa.Table.from_pylist(
        [{"id": row, "name": f"name-{row}"} for row in [1, 2, 4, 5]],
        schema=SCHEMA,
    )


def test_copy_on_write_is_still_the_default(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_copy_on_write_default", merge_on_read=False)
    (data_file,) = _data_files(table, DataFileContent.DATA)

    table.delete("id = 3")

    assert _data_files(table, DataFileContent.POSITION_DELETES) == []
    assert [file.file_path for file in _data_files(table, DataFileContent.DATA)] != [data_file.file_path]


def test_delete_spanning_multiple_data_files(catalog: Catalog) -> None:
    table = _create_table(catalog, "default.test_merge_on_read_multiple_files")
    table.append(
        pa.Table.from_pylist(
            [{"id": row, "name": f"name-{row}"} for row in range(6, 11)],
            schema=SCHEMA,
        )
    )
    assert len(_data_files(table, DataFileContent.DATA)) == 2

    table.delete("id = 2 or id = 8")

    # One delete file per data file it applies to
    assert len(_data_files(table, DataFileContent.POSITION_DELETES)) == 2
    assert len(_data_files(table, DataFileContent.DATA)) == 2
    assert table.scan().to_arrow().sort_by("id") == pa.Table.from_pylist(
        [{"id": row, "name": f"name-{row}"} for row in [1, 3, 4, 5, 6, 7, 9, 10]],
        schema=SCHEMA,
    )


def test_delete_on_a_partitioned_table(catalog: Catalog) -> None:
    identifier = "default.test_merge_on_read_partitioned"
    table = catalog.create_table(
        identifier,
        SCHEMA,
        properties={"format-version": "2", TableProperties.DELETE_MODE: TableProperties.DELETE_MODE_MERGE_ON_READ},
    )
    with table.update_spec() as update_spec:
        update_spec.add_identity("name")
    table.append(
        pa.Table.from_pylist(
            [{"id": row, "name": "a" if row % 2 else "b"} for row in range(1, 7)],
            schema=SCHEMA,
        )
    )

    table.delete("id = 3")

    (delete_file,) = _data_files(table, DataFileContent.POSITION_DELETES)
    (matched_data_file,) = [
        data_file for data_file in _data_files(table, DataFileContent.DATA) if data_file.partition == delete_file.partition
    ]
    assert delete_file.spec_id == matched_data_file.spec_id
    assert table.scan().to_arrow().sort_by("id") == pa.Table.from_pylist(
        [{"id": row, "name": "a" if row % 2 else "b"} for row in [1, 2, 4, 5, 6]],
        schema=SCHEMA,
    )


def test_overwrite_does_not_rewrite_the_data_files(catalog: Catalog) -> None:
    """The incremental-sink case: an overwrite delegates to delete, which now stays merge-on-read."""
    table = _create_table(catalog, "default.test_merge_on_read_overwrite")
    (data_file,) = _data_files(table, DataFileContent.DATA)

    table.overwrite(
        pa.Table.from_pylist([{"id": 3, "name": "updated"}], schema=SCHEMA),
        overwrite_filter="id = 3",
    )

    assert data_file.file_path in [file.file_path for file in _data_files(table, DataFileContent.DATA)]
    assert len(_data_files(table, DataFileContent.POSITION_DELETES)) == 1
    assert table.scan().to_arrow().sort_by("id") == pa.Table.from_pylist(
        [{"id": 1, "name": "name-1"}, {"id": 2, "name": "name-2"}, {"id": 3, "name": "updated"}]
        + [{"id": row, "name": f"name-{row}"} for row in [4, 5]],
        schema=SCHEMA,
    )
