#!/bin/bash

# Sync easy-pytorch with the upstream PyTorch main branch

set -e  # Exit immediately if a command fails

echo "Fetching latest 'main' branch from upstream..."
git fetch upstream main:upstream-main

echo "Switching to your local 'main' branch..."
git checkout main

echo "Merging upstream 'main' into your local 'main'..."
git merge upstream-main

echo "Merge completed. Your 'main' is now up to date with the original PyTorch."
