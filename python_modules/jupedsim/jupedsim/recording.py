# SPDX-License-Identifier: LGPL-3.0-or-later
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import shapely

from jupedsim.internal.aabb import AABB
from jupedsim.sqlite_serialization import update_database_to_latest_version


def open_recording(path: str | Path):
    """Open a trajectory recording, auto-detecting SQLite vs HDF5 format.

    Returns either a :class:`Recording` (SQLite) or :class:`HDF5Recording`.
    Both expose the same API used by the visualizer.
    """
    import h5py

    if h5py.is_hdf5(path):
        return HDF5Recording(str(path))
    return Recording(str(path))


@dataclass
class RecordingAgent:
    """Data for a single agent at a single frame."""

    id: int
    position: tuple[float, float]


@dataclass
class RecordingFrame:
    """A single frame from the simulation."""

    index: int
    agents: list[RecordingAgent]


class Recording:
    __supported_database_version = 3
    """Provides access to a simulation recording in a sqlite database"""

    def __init__(self, db_connection_str: str, uri=False) -> None:
        self.db = sqlite3.connect(
            db_connection_str, uri=uri, isolation_level=None
        )
        update_database_to_latest_version(self.db)
        self._check_version_compatible()

    def frame(self, index: int) -> RecordingFrame:
        """Access a single frame of the recording.

        Arguments:
            index (int): index of the frame to access.

        Returns:
            A single frame.

        """

        def agent_row(cursor, row):
            return RecordingAgent(row[0], (row[1], row[2]))

        cur = self.db.cursor()
        cur.row_factory = agent_row
        res = cur.execute(
            "SELECT id, pos_x, pos_y FROM trajectory_data WHERE frame == (?) ORDER BY id ASC",
            (index,),
        )
        return RecordingFrame(index, res.fetchall())

    def geometry(self) -> shapely.GeometryCollection:
        """Access this recordings' geometry.

        Returns:
            walkable area of the simulation that created this recording.

        """
        cur = self.db.cursor()
        res = cur.execute("SELECT wkt FROM geometry")
        geometries = [shapely.from_wkt(s) for s in res.fetchall()]
        return shapely.union_all(geometries)

    def geometry_id_for_frame(self, frame_id) -> int:
        cur = self.db.cursor()
        res = cur.execute(
            "SELECT geometry_hash from frame_data WHERE frame == ?",
            (frame_id,),
        )
        return res.fetchone()[0]

    def bounds(self) -> AABB:
        cur = self.db.cursor()

        def get_float_or_none(key):
            res = cur.execute(
                f"SELECT value FROM metadata WHERE key == '{key}'"
            ).fetchone()
            return float(res[0]) if res else None

        xmin = get_float_or_none("xmin")
        xmax = get_float_or_none("xmax")
        ymin = get_float_or_none("ymin")
        ymax = get_float_or_none("ymax")

        if None in (xmin, xmax, ymin, ymax):
            raise Exception(
                "Recording has no position bounds metadata. It is likely empty."
            )

        return AABB(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    @property
    def num_frames(self) -> int:
        """Access the number of frames stored in this recording.

        Returns:
            Number of frames in this recording.

        """
        cur = self.db.cursor()
        res = cur.execute("SELECT count(*) FROM frame_data")
        return res.fetchone()[0]

    @property
    def every_nth_frame(self) -> int:
        """How many simulation iterations are skipped between recorded frames.

        Returns:
            every_nth_frame value used when the recording was created.
            Falls back to 1 for recordings that pre-date this metadata field.
        """
        cur = self.db.cursor()
        res = cur.execute(
            "SELECT value FROM metadata WHERE key == 'every_nth_frame'"
        )
        row = res.fetchone()
        return int(row[0]) if row else 1

    @property
    def fps(self) -> float:
        """How many frames are stored per second.

        Returns:
            Frames per second of this recording.

        """
        cur = self.db.cursor()
        res = cur.execute("SELECT value from metadata WHERE key == 'fps'")
        return float(res.fetchone()[0])

    def _check_version_compatible(self) -> None:
        cur = self.db.cursor()
        res = cur.execute("SELECT value FROM metadata WHERE key == 'version'")
        version_string = res.fetchone()[0]
        try:
            version_in_database = int(version_string)
            if version_in_database != self.__supported_database_version:
                raise Exception(
                    f"Incompatible database version. The database supplied is version {version_in_database}. "
                    f"This Program supports version {self.__supported_database_version}"
                )
        except ValueError:
            raise Exception(
                f"Database error, metadata version not an integer. Value found: {version_string}"
            )


class HDF5Recording:
    """Read a trajectory recording written by :class:`~jupedsim.Hdf5TrajectoryWriter`.

    Exposes the same API as :class:`Recording` so the visualizer's
    :class:`~jupedsim_visualizer.replay_widget.ReplayWidget` can consume
    either format without modification.
    """

    def __init__(self, path: str) -> None:
        import h5py
        import numpy as np

        self._h5 = h5py.File(path, "r")
        ds = self._h5["trajectory"]
        self._frames: "np.ndarray" = ds["frame"][:]
        self._ids: "np.ndarray" = ds["id"][:]
        self._xs: "np.ndarray" = ds["x"][:]
        self._ys: "np.ndarray" = ds["y"][:]
        self._num_frames: int = (
            int(self._frames.max()) + 1 if len(self._frames) else 0
        )
        self._fps: float = float(self._h5.attrs.get("fps", 1.0))
        self._every_nth_frame: int = int(
            self._h5.attrs.get("every_nth_frame", 1)
        )
        self._wkt: str = str(self._h5.attrs.get("wkt_geometry", ""))

        # Build per-frame index for O(log n) lookups: sorted unique frame ids
        unique_frames = np.unique(self._frames)
        # Map frame_index → slice of rows
        self._frame_start: dict[int, int] = {}
        self._frame_end: dict[int, int] = {}
        for f in unique_frames:
            lo = int(np.searchsorted(self._frames, f, side="left"))
            hi = int(np.searchsorted(self._frames, f, side="right"))
            self._frame_start[int(f)] = lo
            self._frame_end[int(f)] = hi

    def __del__(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def frame(self, index: int) -> RecordingFrame:
        lo = self._frame_start.get(index)
        hi = self._frame_end.get(index)
        if lo is None:
            return RecordingFrame(index, [])
        agents = [
            RecordingAgent(
                int(self._ids[i]), (float(self._xs[i]), float(self._ys[i]))
            )
            for i in range(lo, hi)
        ]
        return RecordingFrame(index, agents)

    def geometry(self) -> shapely.GeometryCollection:
        return shapely.from_wkt(self._wkt)

    def bounds(self) -> AABB:
        attrs = self._h5.attrs
        try:
            return AABB(
                xmin=float(attrs["xmin"]),
                xmax=float(attrs["xmax"]),
                ymin=float(attrs["ymin"]),
                ymax=float(attrs["ymax"]),
            )
        except KeyError:
            raise Exception(
                "HDF5 recording has no bounding-box metadata. The file may be incomplete."
            )

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def every_nth_frame(self) -> int:
        return self._every_nth_frame
