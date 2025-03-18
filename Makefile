# Dynamically detect the project name based on the top-level directory
PROJECT_NAME := $(notdir $(shell pwd))

# Install dependencies
install:
    pip install -r requirements.txt

# Run tests using pytest
test:
    pytest

# Run tests and generate a coverage report dynamically using the project name
test-coverage:
    pytest --cov=$(PROJECT_NAME) --cov-report=term-missing

# Format code using black for all Python files
format:
    black $(shell find . -name "*.py")

# Remove .pyc files (optional)
clean:
    find . -name "*.pyc" -exec rm -f {} \;
