import pickle
from pathlib import Path
from typing import Any


class FileModelStorage:
    """Stores the pickled model and raw dataset for a workflow on the filesystem.

    MongoDB's 16MB BSON document limit is easy to hit once the pickled model or
    the accumulated dataset arrays get large; both grow without bound over a
    workflow's lifetime, so they live under ``base_directory/<workflow_id>/``
    instead of in the workflow's Mongo document.
    """

    _MODEL_FILENAME = 'model.pkl'
    """Location of the model, which is just a pickled object."""
    _DATASET_FILENAME = 'dataset.tsv'
    """Location of dataset_x + dataset_y.

    This is a plaintext file which is only ever appended to or written, never truncated. A line consists of:
      - the dataset_x values at the given index, space-separated
      - A tab character
      - the dataset_y values at the given index, space-separated
    """

    def __init__(self, base_directory: Path) -> None:
        self._base_directory = base_directory

    def _workflow_dir(self, workflow_id: str) -> Path:
        workflow_dir = self._base_directory / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir

    def write_model(self, workflow_id: str, model: Any) -> None:
        """Pickle and atomically overwrite the model file for a workflow."""
        workflow_dir = self._workflow_dir(workflow_id)
        tmp_path = workflow_dir / f'{self._MODEL_FILENAME}.tmp'
        tmp_path.write_bytes(pickle.dumps(model, protocol=5))
        tmp_path.replace(workflow_dir / self._MODEL_FILENAME)

    def read_model(self, workflow_id: str) -> Any:
        model_bytes = (self._base_directory / workflow_id / self._MODEL_FILENAME).read_bytes()
        return pickle.loads(model_bytes)  # noqa: S301 (trusted - only this service writes to base_directory)

    @staticmethod
    def _format_row(next_x: list[float], next_y: float | list[float]) -> str:
        y_values = next_y if isinstance(next_y, list) else [next_y]
        x_str = ' '.join(str(v) for v in next_x)
        y_str = ' '.join(str(v) for v in y_values)
        return f'{x_str}\t{y_str}\n'

    def append_dataset(
        self, workflow_id: str, next_x: list[float], next_y: float | list[float]
    ) -> None:
        self.append_dataset_batch(workflow_id, [next_x], [next_y])

    def append_dataset_batch(
        self,
        workflow_id: str,
        next_x_list: list[list[float]],
        next_y_list: list[float | list[float]],
    ) -> None:
        workflow_dir = self._workflow_dir(workflow_id)
        rows = ''.join(
            self._format_row(next_x, next_y)
            for next_x, next_y in zip(next_x_list, next_y_list, strict=True)
        )
        with (workflow_dir / self._DATASET_FILENAME).open('a') as f:
            f.write(rows)

    def read_dataset(self, workflow_id: str) -> tuple[list[list[float]], list[float | list[float]]]:
        dataset_path = self._base_directory / workflow_id / self._DATASET_FILENAME
        dataset_x: list[list[float]] = []
        dataset_y: list[float | list[float]] = []
        with dataset_path.open('r') as f:
            for line in f:
                x_part, y_part = line.rstrip('\n').split('\t')
                dataset_x.append([float(v) for v in x_part.split()])
                y_values = [float(v) for v in y_part.split()]
                # collapse dim_y=1 rows back to a bare float, so appending a scalar next_y
                # (see dial_service.py) doesn't produce a mixed float/list dataset_y
                dataset_y.append(y_values[0] if len(y_values) == 1 else y_values)
        return dataset_x, dataset_y
