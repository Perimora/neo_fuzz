"""End-to-end tests for the application."""

import subprocess
import sys
from pathlib import Path


class TestAppE2E:
    """End-to-end tests for the entire application."""

    def test_app_runs_successfully(self) -> None:
        """Test that the app runs without errors."""
        result = subprocess.run(
            ["python3", "app.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "Hello, World!" in result.stdout
        assert "Hello, Stranger!" in result.stdout

    def test_app_creates_log_files(self) -> None:
        """Test that the app creates log files."""
        # Run the app
        subprocess.run(
            ["python3", "app.py"], capture_output=True, cwd=Path(__file__).parent.parent.parent
        )

        # Check that log files are created
        log_dir = Path(__file__).parent.parent.parent / "logs" / "development"
        assert log_dir.exists()
        assert (log_dir / "master.log").exists()
        assert (log_dir / "info.log").exists()
        assert (log_dir / "warning.log").exists()

    def test_pytest_runs_successfully(self) -> None:
        """Test that pytest runs all tests successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test/unit", "-v"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "PASSED" in result.stdout
