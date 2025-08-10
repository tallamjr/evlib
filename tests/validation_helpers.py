"""
Event data validation using Pandera schemas.

This module provides validation for event camera data using pandera schemas.
Primarily intended for testing and development use.

Usage:
    from evlib.validation import quick_validate_events, validate_events

    # Quick validation for obvious errors
    is_valid = quick_validate_events(events_df)

    # Detailed validation with reporting
    result = validate_events(events_df, sensor_type="prophesee_gen4")
"""

import polars as pl
import pandera.polars as pa
from pandera import Field
from typing import Dict, Any, Union, Optional, Tuple
import logging
from pathlib import Path

# Configure logging for validation module
logger = logging.getLogger(__name__)

# Event camera sensor constraints (based on common sensor specifications)
SENSOR_CONSTRAINTS = {
    # Common event camera resolutions
    "prophesee_gen4": {"max_x": 1279, "max_y": 719, "width": 1280, "height": 720},  # 1280x720
    "prophesee_gen3": {"max_x": 639, "max_y": 479, "width": 640, "height": 480},  # 640x480
    "davis346": {"max_x": 345, "max_y": 239, "width": 346, "height": 240},  # 346x240
    "davis640": {"max_x": 639, "max_y": 479, "width": 640, "height": 480},  # 640x480
    "etram": {"max_x": 1279, "max_y": 719, "width": 1280, "height": 720},  # eTram dataset
    "generic_hd": {"max_x": 1279, "max_y": 719, "width": 1280, "height": 720},  # HD resolution
    "generic_large": {"max_x": 9999, "max_y": 9999, "width": 10000, "height": 10000},  # Very large sensors
}


def create_event_schema(
    sensor_type: str = "generic_large",
    strict_timestamps: bool = False,  # Changed default to False
    allow_negative_timestamps: bool = False,
    data_format: str = "duration",  # "duration" for Duration type, "float" for seconds
    polarity_encoding: str = "minus_one_one",  # "minus_one_one" or "zero_one"
) -> pa.DataFrameSchema:
    """
    Create a Pandera validation schema for event data.

    Args:
        sensor_type: Type of sensor ('prophesee_gen4', 'davis346', 'etram', etc.)
        strict_timestamps: Whether to enforce monotonic timestamps (note: disabled for refractory filtering)
        allow_negative_timestamps: Whether to allow negative timestamp values
        data_format: Format of timestamp data ("duration" or "float")

    Returns:
        Pandera LazyFrameSchema for validating event data

    Note:
        This schema matches evlib's internal data format:
        - x, y: Int16 coordinates
        - timestamp: Duration in microseconds OR Float64 in seconds
        - polarity: Int8 with values [-1, 1] (not 0)
    """
    # Get sensor constraints
    constraints = SENSOR_CONSTRAINTS.get(sensor_type, SENSOR_CONSTRAINTS["generic_large"])
    max_x, max_y = constraints["max_x"], constraints["max_y"]

    # Create schema based on data format
    if data_format == "duration":
        # Duration format (evlib's internal format)
        timestamp_column = pa.Column(
            pl.Duration,
            nullable=False,
            description="Event timestamp as Duration in microseconds",
        )
    else:
        # Float format (for validation and analysis)
        min_timestamp = -1e18 if allow_negative_timestamps else 0
        max_timestamp = 1e18
        timestamp_column = pa.Column(
            pl.Float64,
            checks=[
                pa.Check.greater_than_or_equal_to(min_timestamp),
                pa.Check.less_than_or_equal_to(max_timestamp),
            ],
            nullable=False,
            description="Event timestamp in seconds",
        )

    schema = pa.DataFrameSchema(
        {
            # X coordinate: Int16 for memory efficiency
            "x": pa.Column(
                pl.Int16,
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(max_x),
                ],
                nullable=False,
                description=f"X coordinate (0 to {max_x} for {sensor_type})",
            ),
            # Y coordinate: Int16 for memory efficiency
            "y": pa.Column(
                pl.Int16,
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(max_y),
                ],
                nullable=False,
                description=f"Y coordinate (0 to {max_y} for {sensor_type})",
            ),
            # Timestamp: format-dependent - API uses 't' column name
            "t": timestamp_column,
            # Polarity: Int8 with encoding-specific values
            "polarity": pa.Column(
                pl.Int8,
                checks=[
                    pa.Check.isin([-1, 1] if polarity_encoding == "minus_one_one" else [0, 1]),
                ],
                nullable=False,
                description=f"Event polarity ({polarity_encoding} encoding)",
            ),
        },
        strict=True,  # Only allow specified columns
        coerce=False,  # Don't coerce - expect exact types
    )

    return schema


def create_raw_event_schema(data_format: str = "float") -> pa.DataFrameSchema:
    """
    Create a validation schema for raw event data (before processing).
    This is more lenient to catch obvious garbage while allowing valid edge cases.

    Args:
        data_format: Format of timestamp data ("duration" or "float")
    """
    if data_format == "duration":
        timestamp_column = pa.Column(
            pl.Duration,
            nullable=False,
            description="Raw timestamp as Duration (very permissive)",
        )
    else:
        timestamp_column = pa.Column(
            pl.Float64,
            checks=[
                pa.Check.greater_than_or_equal_to(-1e18),
                pa.Check.less_than_or_equal_to(1e18),
            ],
            nullable=False,
            description="Raw timestamp in seconds (very permissive range)",
        )

    return pa.DataFrameSchema(
        {
            "x": pa.Column(
                pl.Int16,  # Match evlib format
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(65535),  # 16-bit max
                ],
                nullable=False,
                description="Raw X coordinate (16-bit unsigned)",
            ),
            "y": pa.Column(
                pl.Int16,  # Match evlib format
                checks=[
                    pa.Check.greater_than_or_equal_to(0),
                    pa.Check.less_than_or_equal_to(65535),  # 16-bit max
                ],
                nullable=False,
                description="Raw Y coordinate (16-bit unsigned)",
            ),
            "t": timestamp_column,
            "polarity": pa.Column(
                pl.Int8,  # Match evlib format
                checks=[
                    pa.Check.greater_than_or_equal_to(-10),  # Catch obvious garbage
                    pa.Check.less_than_or_equal_to(10),
                ],
                nullable=False,
                description="Raw polarity (permissive range to catch garbage)",
            ),
        },
        strict=True,
        coerce=False,  # Don't coerce - expect correct types
    )


def validate_events(
    events_df: pl.LazyFrame,
    sensor_type: str = "generic_large",
    strict: bool = True,
    data_format: str = None,
    sample_size: Optional[int] = None,
    polarity_encoding: str = "minus_one_one",
) -> Dict[str, Any]:
    """
    Validate event data and return validation results.

    Args:
        events_df: Polars LazyFrame with event data
        sensor_type: Type of sensor for constraint validation
        strict: Whether to use strict validation
        data_format: Format of timestamp data ("duration" or "float"), auto-detected if None
        sample_size: Number of events to sample for large datasets (None = validate all)

    Returns:
        Dictionary with validation results and statistics
    """
    validation_results = {"valid": False, "errors": [], "warnings": [], "statistics": {}}

    try:
        # Convert LazyFrame to DataFrame for validation (Pandera issue with LazyFrames)
        if hasattr(events_df, "collect"):
            # Sample for large datasets if requested
            if sample_size:
                df_to_validate = events_df.limit(sample_size).collect()
                validation_results["warnings"].append(
                    f"Validation performed on sample of {sample_size} events"
                )
            else:
                df_to_validate = events_df.collect()
        else:
            df_to_validate = events_df

        # Auto-detect data format if not specified
        if data_format is None:
            timestamp_dtype = df_to_validate["t"].dtype
            if timestamp_dtype == pl.Duration:
                data_format = "duration"
            else:
                data_format = "float"

        # Create appropriate schema
        if strict:
            schema = create_event_schema(
                sensor_type, data_format=data_format, polarity_encoding=polarity_encoding
            )
        else:
            schema = create_raw_event_schema(data_format=data_format)

        # Attempt validation
        validated_df = schema.validate(df_to_validate)

        # If we get here, validation passed
        validation_results["valid"] = True

        # Collect statistics
        validation_results["statistics"] = _collect_event_statistics(validated_df.lazy())

        # Additional quality checks
        quality_warnings = _check_data_quality(validated_df.lazy())
        validation_results["warnings"].extend(quality_warnings)

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


def _collect_event_statistics(events_df: pl.LazyFrame) -> Dict[str, Any]:
    """Collect comprehensive statistics from event data."""
    try:
        # Handle Duration vs Float timestamps differently
        timestamp_dtype = events_df.select(pl.col("t")).collect().dtypes[0]

        if timestamp_dtype == pl.Duration:
            # Convert Duration to seconds for statistics
            timestamp_expr = pl.col("t").dt.total_microseconds() / 1_000_000
        else:
            timestamp_expr = pl.col("t")

        stats_df = events_df.select(
            [
                pl.len().alias("event_count"),
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
                timestamp_expr.min().alias("timestamp_min"),
                timestamp_expr.max().alias("timestamp_max"),
                pl.col("polarity").n_unique().alias("unique_polarity_count"),
                (pl.col("polarity") == 1).sum().alias("positive_events"),
                (pl.col("polarity") == -1).sum().alias("negative_events"),
            ]
        ).collect()

        event_count = stats_df["event_count"][0]
        positive_events = stats_df["positive_events"][0]
        negative_events = stats_df["negative_events"][0]

        return {
            "event_count": event_count,
            "coordinate_ranges": {
                "x": (stats_df["x_min"][0], stats_df["x_max"][0]),
                "y": (stats_df["y_min"][0], stats_df["y_max"][0]),
            },
            "timestamp_range": (stats_df["timestamp_min"][0], stats_df["timestamp_max"][0]),
            "duration_seconds": stats_df["timestamp_max"][0] - stats_df["timestamp_min"][0],
            "unique_polarity_count": stats_df["unique_polarity_count"][0],
            "polarity_distribution": {
                "positive": positive_events,
                "negative": negative_events,
                "positive_ratio": positive_events / event_count if event_count > 0 else 0,
            },
            "event_rate_hz": (
                event_count / (stats_df["timestamp_max"][0] - stats_df["timestamp_min"][0])
                if stats_df["timestamp_max"][0] > stats_df["timestamp_min"][0]
                else 0
            ),
        }
    except Exception as e:
        logger.warning(f"Failed to collect statistics: {e}")
        return {"error": str(e)}


def _check_data_quality(events_df: pl.LazyFrame) -> list:
    """Check for common data quality issues and return warnings."""
    warnings = []

    try:
        # Check for timestamp monotonicity (informational only)
        timestamp_dtype = events_df.select(pl.col("t")).collect().dtypes[0]

        if timestamp_dtype == pl.Duration:
            backward_jumps = (
                events_df.select((pl.col("t").diff() < pl.duration(microseconds=0)).sum()).collect().row(0)[0]
            )
        else:
            backward_jumps = events_df.select((pl.col("t").diff() < 0).sum()).collect().row(0)[0]

        if backward_jumps > 0:
            warnings.append(
                f"Found {backward_jumps} backward timestamp jumps (may be expected after refractory filtering)"
            )

        # Check polarity balance
        stats = events_df.select(
            [
                (pl.col("polarity") == 1).sum().alias("positive"),
                (pl.col("polarity") == -1).sum().alias("negative"),
                pl.len().alias("total"),
            ]
        ).collect()

        positive_ratio = stats["positive"][0] / stats["total"][0]
        if positive_ratio < 0.1 or positive_ratio > 0.9:
            warnings.append(f"Extreme polarity imbalance: {positive_ratio:.1%} positive events")

        # Check for spatial clustering (potential hot pixels)
        pixel_counts = (
            events_df.group_by([pl.col("x"), pl.col("y")])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .limit(10)
            .collect()
        )

        if len(pixel_counts) > 0:
            max_pixel_events = pixel_counts["count"][0]
            total_events = stats["total"][0]
            if max_pixel_events > total_events * 0.1:  # Single pixel has >10% of events
                warnings.append(
                    f"Potential hot pixel detected: single pixel has {max_pixel_events} events ({max_pixel_events/total_events:.1%} of total)"
                )

    except Exception as e:
        warnings.append(f"Quality check failed: {e}")

    return warnings


def quick_validate_events(events_df: pl.LazyFrame, sample_size: int = 10000) -> bool:
    """
    Quick validation check for event data.

    Args:
        events_df: Polars LazyFrame with event data
        sample_size: Number of events to sample for validation

    Returns:
        True if data passes basic validation, False otherwise
    """
    try:
        # Convert LazyFrame to DataFrame for validation (Pandera issue with LazyFrames)
        if hasattr(events_df, "collect"):
            df_to_validate = events_df.limit(sample_size).collect()
        else:
            df_to_validate = events_df

        # Auto-detect data format
        timestamp_dtype = df_to_validate["t"].dtype
        data_format = "duration" if timestamp_dtype == pl.Duration else "float"

        raw_schema = create_raw_event_schema(data_format=data_format)
        raw_schema.validate(df_to_validate)
        return True
    except Exception:
        return False


# =============================================================================
# Pre-defined schemas for common use cases
# =============================================================================

PROPHESEE_GEN4_SCHEMA = create_event_schema("prophesee_gen4", data_format="duration")
DAVIS_SCHEMA = create_event_schema("davis346", data_format="duration")
ETRAM_SCHEMA = create_event_schema("etram", data_format="duration")
PERMISSIVE_SCHEMA = create_raw_event_schema(data_format="duration")
