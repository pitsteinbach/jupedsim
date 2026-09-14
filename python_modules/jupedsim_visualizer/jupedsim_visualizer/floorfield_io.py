# SPDX-License-Identifier: LGPL-3.0-or-later
"""HDF5 serialization for floor-field grids."""

from __future__ import annotations


def save_travel_times(
    path: str,
    field: dict,
    destination: tuple[float, float],
    *,
    append: bool = False,
    speed_field: dict | None = None,
) -> None:
    """Write a travel-time grid to an HDF5 file.

    The file layout is::

        /speed_field/data          float32 [height × width]  (written once)
        /travel_times/<dest>/data  float32 [height × width]  (one per dest)

    where ``<dest>`` is ``x{dest_x:.4f}_y{dest_y:.4f}``.

    Each dataset carries attrs: origin_x, origin_y, cell_size, width, height.
    Travel-time datasets additionally carry destination_x, destination_y and
    units="seconds".

    Args:
        path:        Output file path.
        field:       Dict from ``Floorfield.travel_times()``
                     (keys: data, width, height, origin, cell_size).
        destination: (x, y) world-space destination used to compute the field.
        append:      Open in append mode so multiple destinations accumulate.
        speed_field: Optional dict from ``Floorfield.speed_field()``; written
                     once under /speed_field when provided and not yet present.
    """
    import h5py
    import numpy as np

    w, h = field["width"], field["height"]
    ox, oy = field["origin"]
    cs: float = field["cell_size"]

    mode = "a" if append else "w"
    with h5py.File(path, mode) as f:
        if speed_field is not None and "speed_field" not in f:
            sw, sh = speed_field["width"], speed_field["height"]
            sox, soy = speed_field["origin"]
            scs: float = speed_field["cell_size"]
            grp = f.create_group("speed_field")
            ds = grp.create_dataset(
                "data",
                data=np.array(speed_field["data"], dtype=np.float32).reshape(
                    sh, sw
                ),
                compression="gzip",
            )
            ds.attrs["origin_x"] = sox
            ds.attrs["origin_y"] = soy
            ds.attrs["cell_size"] = scs
            ds.attrs["width"] = sw
            ds.attrs["height"] = sh

        dest_key = f"x{destination[0]:.4f}_y{destination[1]:.4f}"
        grp_path = f"travel_times/{dest_key}"
        if grp_path in f:
            del f[grp_path]
        grp = f.create_group(grp_path)
        ds = grp.create_dataset(
            "data",
            data=np.array(field["data"], dtype=np.float32).reshape(h, w),
            compression="gzip",
        )
        ds.attrs["origin_x"] = ox
        ds.attrs["origin_y"] = oy
        ds.attrs["cell_size"] = cs
        ds.attrs["width"] = w
        ds.attrs["height"] = h
        ds.attrs["destination_x"] = destination[0]
        ds.attrs["destination_y"] = destination[1]
        ds.attrs["units"] = "seconds"


def load_travel_times(
    path: str, dest_key: str | None = None
) -> dict | dict[str, dict]:
    """Load travel-time grid(s) from an HDF5 file.

    If *dest_key* is given return one ``{data, width, height, origin,
    cell_size, destination}`` dict.  Otherwise return ``{dest_key: dict}``
    for every destination stored in the file.
    """
    import h5py

    with h5py.File(path, "r") as f:
        tt_grp = f.get("travel_times", {})

        def _read(grp) -> dict:
            ds = grp["data"]
            return {
                "data": ds[:].flatten().tolist(),
                "width": int(ds.attrs["width"]),
                "height": int(ds.attrs["height"]),
                "origin": (
                    float(ds.attrs["origin_x"]),
                    float(ds.attrs["origin_y"]),
                ),
                "cell_size": float(ds.attrs["cell_size"]),
                "destination": (
                    float(ds.attrs["destination_x"]),
                    float(ds.attrs["destination_y"]),
                ),
            }

        if dest_key is not None:
            return _read(tt_grp[dest_key])
        return {k: _read(tt_grp[k]) for k in tt_grp}
