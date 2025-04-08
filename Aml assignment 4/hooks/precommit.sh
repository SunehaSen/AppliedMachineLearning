#!/bin/bash

# Get the current branch name
BRANCH=$(git symbolic-ref --short HEAD)

# Only run tests when committing to main branch
if [ "$BRANCH" = "main" ]; then
    echo "Running tests before committing to main branch..."
    
    # Run the tests with pytest
    python -m pytest test.py -v
    
    # Check the exit code of pytest
    if [ $? -ne 0 ]; then
        echo "Tests failed! Commit aborted."
        exit 1
    fi
    
    echo "All tests passed! Proceeding with commit."
fi

exit 0