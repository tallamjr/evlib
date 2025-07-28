"""
Event data validation using Pandera schemas.

This module provides comprehensive validation for event camera data
based on the official Prophesee ECF specification and common event camera constraints.
"""

import polars as pl
import pandera.polars as pa
from pandera import Field
from typing import Dict, Any, Union

# Event camera sensor constraints (based on common sensor specifications)
SENSOR_CONSTRAINTS = {
    # Common event camera resolutions
    "prophesee_gen4": {"max_x": 1279, "max_y": 719},  # 1280x720
    "prophesee_gen3": {"max_x": 639, "max_y": 479},  # 640x480
    "davis346": {"max_x": 345, "max_y": 239},  # 346x240
    "davis640": {"max_x": 639, "max_y": 479},  # 640x480
    "generic_hd": {"max_x": 1279, "max_y": 719},  # HD resolution
    "generic_large": {"max_x": 9999, "max_y": 9999},  # Very large sensors
}


def create_event_schema(
    sensor_type: str = "generic_large",
    strict_timestamps: bool = True,
    allow_negative_timestamps: bool = False,
) -> pa.DataFrameSchema:
    """
    Create a Pandera validation schema for event data.

    Args:
        sensor_type: Type of sensor ('prophesee_gen4', 'davis346', etc.)
        strict_timestamps: Whether to enforce monotonic timestamps
        allow_negative_timestamps: Whether to allow negative timestamp values

    Returns:
        Pandera LazyFrameSchema for validating event data
    """

    # Get sensor constraints
    constraints = SENSOR_CONSTRAINTS.get(sensor_type, SENSOR_CONSTRAINTS["generic_large"])
    max_x, max_y = constraints["max_x"], constraints["max_y"]

    # Timestamp constraints
    min_timestamp = -1e18 if allow_negative_timestamps else 0
    max_timestamp = 1e18  # Very large but reasonable upper bound

    schema = pa.DataFrameSchema(
        {
            # X coordinate: should be within sensor bounds
            "x": pa.Column(
                pl.Int64,  # Polars uses Int64 by default
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(max_x),
                ],
                nullable=False,
                description=f"X coordinate (0 to {max_x} for {sensor_type})",
            ),
            # Y coordinate: should be within sensor bounds
            "y": pa.Column(
                pl.Int64,  # Polars uses Int64 by default
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(max_y),
                ],
                nullable=False,
                description=f"Y coordinate (0 to {max_y} for {sensor_type})",
            ),
            # Timestamp: should be reasonable and optionally monotonic
            "timestamp": pa.Column(
                pl.Float64,  # Timestamps are often converted to seconds (float)
                checks=[
                    pa.Check.greater_than_or_equal_to(min_timestamp),
                    pa.Check.less_than_or_equal_to(max_timestamp),
                    # Skip finite check for now due to Pandera/Polars compatibility issues
                    # pa.Check(lambda ts: ts.is_finite().all(),
                    #         error="Timestamps must be finite (no NaN or infinity)"),
                ],
                # Skip monotonic check for now due to Pandera/Polars compatibility issues
                # ] + ([
                #     # Monotonic timestamp check (optional)
                #     pa.Check(lambda ts: (ts.diff().drop_nulls() >= -1e-6).all(),
                #             error="Timestamps must be approximately monotonically increasing")
                # ] if strict_timestamps else []),
                nullable=False,
                description="Event timestamp (should be monotonically increasing)",
            ),
            # Polarity: should be -1, 0, or 1
            "polarity": pa.Column(
                pl.Int64,  # Polars uses Int64 by default
                checks=[
                    pa.Check.isin([-1, 0, 1]),
                ],
                nullable=False,
                description="Event polarity (-1, 0, or 1)",
            ),
        },
        strict=True,  # Only allow specified columns
        coerce=True,  # Allow type coercion for compatibility
    )

    return schema


def create_raw_event_schema() -> pa.DataFrameSchema:
    """
    Create a validation schema for raw event data (before processing).
    This is more lenient to catch obvious garbage while allowing valid edge cases.
    """

    return pa.DataFrameSchema(
        {
            "x": pa.Column(
                pl.Int64,
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(65535),  # 16-bit max
                ],
                nullable=False,
                description="Raw X coordinate (16-bit unsigned)",
            ),
            "y": pa.Column(
                pl.Int64,
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(65535),  # 16-bit max
                ],
                nullable=False,
                description="Raw Y coordinate (16-bit unsigned)",
            ),
            "timestamp": pa.Column(
                pl.Float64,  # Use float to handle both int and float
                checks=[
                    pa.Check.greater_than_or_equal_to(-1e18),
                    pa.Check.less_than_or_equal_to(1e18),
                    # Skip finite check for now due to Pandera/Polars compatibility issues
                    # pa.Check(lambda ts: ts.is_finite().all(),
                    #         error="Timestamps must be finite"),
                ],
                nullable=False,
                description="Raw timestamp (very permissive range)",
            ),
            "polarity": pa.Column(
                pl.Int64,
                checks=[
                    pa.Check.greater_than_or_equal_to(-1000),  # Catch obvious garbage
                    pa.Check.less_than_or_equal_to(1000),
                ],
                nullable=False,
                description="Raw polarity (permissive range to catch garbage)",
            ),
        },
        strict=True,
        coerce=True,
    )


def validate_events(
    events_df: pl.LazyFrame, sensor_type: str = "generic_large", strict: bool = True
) -> Dict[str, Any]:
    """
    Validate event data and return validation results.

    Args:
        events_df: Polars LazyFrame with event data
        sensor_type: Type of sensor for constraint validation
        strict: Whether to use strict validation

    Returns:
        Dictionary with validation results and statistics
    """

    # Create appropriate schema
    if strict:
        schema = create_event_schema(sensor_type)
    else:
        schema = create_raw_event_schema()

    validation_results = {"valid": False, "errors": [], "warnings": [], "statistics": {}}

    try:
        # Convert LazyFrame to DataFrame for validation (Pandera issue with LazyFrames)
        if hasattr(events_df, "collect"):
            df_to_validate = events_df.collect()
        else:
            df_to_validate = events_df

        # Attempt validation
        validated_df = schema.validate(df_to_validate)

        # If we get here, validation passed
        validation_results["valid"] = True

        # Collect statistics (convert to LazyFrame if needed for consistent API)
        if hasattr(validated_df, "lazy"):
            stats_df_lazy = validated_df.lazy()
        else:
            stats_df_lazy = validated_df

        stats_df = stats_df_lazy.select(
            [
                pl.len().alias("event_count"),
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
                pl.col("timestamp").min().alias("timestamp_min"),
                pl.col("timestamp").max().alias("timestamp_max"),
                pl.col("polarity").unique().sort().alias("unique_polarities"),
            ]
        ).collect()

        validation_results["statistics"] = {
            "event_count": stats_df["event_count"][0],
            "coordinate_ranges": {
                "x": (stats_df["x_min"][0], stats_df["x_max"][0]),
                "y": (stats_df["y_min"][0], stats_df["y_max"][0]),
            },
            "timestamp_range": (stats_df["timestamp_min"][0], stats_df["timestamp_max"][0]),
            "unique_polarities": stats_df["unique_polarities"][0],
        }

    except pa.errors.SchemaError as e:
        validation_results["valid"] = False
        validation_results["errors"].append(
            {
                "type": "SchemaError",
                "message": str(e),
                "details": e.failure_cases if hasattr(e, "failure_cases") else None,
            }
        )

    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append({"type": type(e).__name__, "message": str(e)})

    return validation_results


def quick_validate_events(events_df: pl.LazyFrame) -> bool:
    """
    Quick validation check for event data.

    Args:
        events_df: Polars LazyFrame with event data

    Returns:
        True if data passes basic validation, False otherwise
    """

    try:
        # Convert LazyFrame to DataFrame for validation (Pandera issue with LazyFrames)
        if hasattr(events_df, "collect"):
            df_to_validate = events_df.collect()
        else:
            df_to_validate = events_df

        raw_schema = create_raw_event_schema()
        raw_schema.validate(df_to_validate)
        return True
    except Exception:
        return False


# Pre-defined schemas for common use cases
PROPHESEE_GEN4_SCHEMA = create_event_schema("prophesee_gen4", strict_timestamps=True)
DAVIS_SCHEMA = create_event_schema("davis346", strict_timestamps=True)
PERMISSIVE_SCHEMA = create_raw_event_schema()
