#!/usr/bin/env python3
"""
Test documentation examples using pytest-markdown-docs.

This script provides a convenient way to test all documentation examples
and generate reports on their status.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_docs_tests(docs_path=None, verbose=False, max_failures=None):
    """Run documentation tests."""

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--markdown-docs",
        "-v" if verbose else "-q",
        "--tb=short",
        "--color=yes",
    ]

    if max_failures:
        cmd.extend(["--maxfail", str(max_failures)])

    if docs_path:
        cmd.append(str(docs_path))
    else:
        cmd.append("docs/")

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def generate_test_report(output_file="docs-test-report.txt"):
    """Generate a detailed test report."""

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--markdown-docs",
        "docs/",
        "-v",
        "--tb=short",
        "--maxfail=50",
        "--no-header",
        "--quiet",
    ]

    print(f"Generating test report: {output_file}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Create report
    report_content = f"""# Documentation Test Report

## Summary
- Command: {' '.join(cmd)}
- Exit code: {result.returncode}
- Tests {'PASSED' if result.returncode == 0 else 'FAILED'}

## Test Output

### STDOUT
```
{result.stdout}
```

### STDERR
```
{result.stderr}
```

## Next Steps

The documentation tests are now set up with pytest-markdown-docs.

### Current Status
- ✅ pytest-markdown-docs is installed and configured
- ✅ Configuration is set up in pyproject.toml
- ✅ Mock evlib module is available for testing
- ✅ Test fixtures handle matplotlib and data files
- ⚠️  Some API functions may not exist yet (this is expected)

### To Fix Failing Tests
1. Implement missing API functions in evlib
2. Update documentation examples to match current API
3. Add more comprehensive mocking for unavailable features

### Running Tests
```bash
# Test all documentation
pytest --markdown-docs docs/

# Test specific file
pytest --markdown-docs docs/getting-started/quickstart.md

# Test with verbose output
pytest --markdown-docs docs/ -v

# Test with limited failures
pytest --markdown-docs docs/ --maxfail=5
```

### CI Integration
The tests are integrated into GitHub Actions and will run automatically on:
- Push to main/master branch
- Pull requests
- Changes to docs/ or python/ directories

### Configuration Files
- pyproject.toml: Main configuration
- docs/conftest.py: Test fixtures and mocking
- .github/workflows/test-docs.yml: CI pipeline
"""

    with open(output_file, "w") as f:
        f.write(report_content)

    print(f"Report generated: {output_file}")
    return result.returncode


def list_testable_docs():
    """List all testable documentation files."""
    docs_dir = Path("docs")

    if not docs_dir.exists():
        print("ERROR: docs/ directory not found")
        return

    md_files = list(docs_dir.glob("**/*.md"))
    print(f"Found {len(md_files)} markdown files:")

    total_code_blocks = 0
    for file_path in sorted(md_files):
        try:
            content = file_path.read_text()
            python_blocks = content.count("```python")
            total_code_blocks += python_blocks

            status = "✅" if python_blocks > 0 else "❌"
            print(f"  {status} {file_path} ({python_blocks} code blocks)")

        except Exception as e:
            print(f"  ❌ {file_path} (Error: {e})")

    print(f"\nTotal: {total_code_blocks} Python code blocks in {len(md_files)} files")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test documentation examples")
    parser.add_argument("--path", help="Path to specific documentation file or directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max-failures", type=int, help="Maximum number of failures before stopping")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--list", action="store_true", help="List testable documentation files")

    args = parser.parse_args()

    if args.list:
        list_testable_docs()
        return

    if args.report:
        return generate_test_report()

    return run_docs_tests(docs_path=args.path, verbose=args.verbose, max_failures=args.max_failures)


if __name__ == "__main__":
    sys.exit(main())
