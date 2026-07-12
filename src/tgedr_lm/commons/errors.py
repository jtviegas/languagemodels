"""Custom exceptions for the commons module."""


class UnknownTrainingArgsError(ValueError):
    """Raised when updating training arguments with unknown member names."""

    def __init__(self, unknown_fields: set[str]) -> None:
        """Initialize the exception with the unknown field names."""
        unknown = ", ".join(sorted(unknown_fields))
        msg = f"Unknown training argument(s): {unknown}"
        super().__init__(msg)
