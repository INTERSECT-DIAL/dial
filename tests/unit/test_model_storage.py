import pytest

from dial_service.model_storage import FileModelStorage

WORKFLOW_ID = 'deadbeef12345678deadbeef'


def test_write_and_read_model_round_trip(tmp_path):
    storage = FileModelStorage(tmp_path)
    storage.write_model(WORKFLOW_ID, {'some': 'pickled', 'state': [1, 2, 3]})
    assert storage.read_model(WORKFLOW_ID) == {'some': 'pickled', 'state': [1, 2, 3]}


def test_write_model_overwrite_is_atomic_replace(tmp_path):
    storage = FileModelStorage(tmp_path)
    storage.write_model(WORKFLOW_ID, {'version': 'first'})
    storage.write_model(WORKFLOW_ID, {'version': 'second', 'extra': 'data'})
    assert storage.read_model(WORKFLOW_ID) == {'version': 'second', 'extra': 'data'}
    # no leftover temp file after a successful replace
    assert not (tmp_path / WORKFLOW_ID / 'model.pkl.tmp').exists()


def test_read_dataset_missing_file_raises(tmp_path):
    storage = FileModelStorage(tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.read_dataset(WORKFLOW_ID)


def test_append_empty_batch_touches_dataset_file(tmp_path):
    """Mirrors initialize_workflow's no-initial-data path: the file must exist but be empty."""
    storage = FileModelStorage(tmp_path)
    storage.append_dataset_batch(WORKFLOW_ID, [], [])
    assert (tmp_path / WORKFLOW_ID / 'dataset.tsv').exists()
    assert storage.read_dataset(WORKFLOW_ID) == ([], [])


def test_append_and_read_dataset_single(tmp_path):
    storage = FileModelStorage(tmp_path)
    storage.append_dataset(WORKFLOW_ID, [1.0, 2.5], 3.5)
    storage.append_dataset(WORKFLOW_ID, [4.0, 5.5], [6.5, 7.5])
    dataset_x, dataset_y = storage.read_dataset(WORKFLOW_ID)
    assert dataset_x == [[1.0, 2.5], [4.0, 5.5]]
    # dim_y=1 rows collapse back to a bare float (see model_storage.py's read_dataset docstring)
    assert dataset_y == [3.5, [6.5, 7.5]]


def test_append_dataset_batch(tmp_path):
    storage = FileModelStorage(tmp_path)
    storage.append_dataset_batch(
        WORKFLOW_ID,
        [[1.0], [2.0], [3.0]],
        [10.0, [20.0, 21.0], 30.0],
    )
    dataset_x, dataset_y = storage.read_dataset(WORKFLOW_ID)
    assert dataset_x == [[1.0], [2.0], [3.0]]
    assert dataset_y == [10.0, [20.0, 21.0], 30.0]


def test_dataset_appends_accumulate_across_calls(tmp_path):
    storage = FileModelStorage(tmp_path)
    storage.append_dataset_batch(WORKFLOW_ID, [[1.0]], [1.0])
    storage.append_dataset(WORKFLOW_ID, [2.0], 2.0)
    dataset_x, dataset_y = storage.read_dataset(WORKFLOW_ID)
    assert dataset_x == [[1.0], [2.0]]
    assert dataset_y == [1.0, 2.0]
