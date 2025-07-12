"""Unit tests for app functions."""

from app import greet


class TestGreetFunction:
    """Test the greet function in isolation."""

    def test_greet_with_name(self) -> None:
        """Test greet function with a name."""
        result = greet("Alice")
        assert result == "Hello, Alice!"

    def test_greet_with_empty_string(self) -> None:
        """Test greet function with empty string."""
        result = greet("")
        assert result == "Hello, Stranger!"

    def test_greet_with_whitespace(self) -> None:
        """Test greet function with whitespace."""
        result = greet("   ")
        assert result == "Hello,    !"
