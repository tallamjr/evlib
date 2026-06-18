"""Reduce a sample decoded by evlib.load_events to a canonical digest."""

import evlib
import polars as pl

from tests.conformance import canonical


def evlib_events(path):
    """Return canonical (x, y, pol, t) tuples from evlib's decode.

    Loads with sort=False (canonical.compute_digest imposes the total order),
    maps evlib's -1/1 polarity to {0, 1}, and converts the Duration timestamp
    to integer microseconds."""
    lf = evlib.load_events(str(path), sort=False)
    df = lf.select(
        pl.col("x").cast(pl.Int64),
        pl.col("y").cast(pl.Int64),
        # evlib polarity is -1/1; canonical form is 0/1.
        ((pl.col("polarity") == 1).cast(pl.Int64)).alias("pol"),
        pl.col("t").dt.total_microseconds().alias("t"),
    ).collect()
    return list(
        zip(
            df["x"].to_list(),
            df["y"].to_list(),
            df["pol"].to_list(),
            df["t"].to_list(),
        )
    )


def evlib_geometry(path):
    """Sensor geometry (max_x+1, max_y+1) observed in the decoded events.

    OpenEB writes the header geometry; evlib does not surface it on the frame,
    so we compare against the observed coordinate extent, which the OpenEB
    digest also reduces to (see openeb_runner)."""
    events = evlib_events(path)
    max_x = max((e[0] for e in events), default=-1)
    max_y = max((e[1] for e in events), default=-1)
    return (max_x + 1, max_y + 1)


def evlib_digest(path):
    return canonical.compute_digest(evlib_events(path), geometry=evlib_geometry(path))
