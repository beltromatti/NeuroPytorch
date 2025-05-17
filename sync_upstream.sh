#!/bin/bash

# Sync easy-pytorch with the upstream PyTorch main branch

set -e  # Exit immediately if a command fails

echo "Fetching 'main' from upstream..."
git fetch upstream

echo "Checking out 'main' branch..."
git checkout main

echo "Rebasing your 'main' on top of 'upstream/main'..."
git rebase upstream/main

echo "Done. Your local 'main' is now exactly aligned with upstream."