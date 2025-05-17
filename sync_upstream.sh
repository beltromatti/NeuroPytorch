#!/bin/bash

# Sync easy-pytorch with the upstream PyTorch main branch

set -e  # Exit immediately if a command fails

echo "Fetching 'main' from upstream..."
git fetch upstream

echo "Checking out 'main' branch..."
git checkout main

echo "Rebasing your 'main' on top of 'upstream/main'..."
git rebase upstream/main

echo "Pushing updates on github..."
git push --force

echo "DONE"

