@echo off
echo Fetching 'main' from upstream...
git fetch upstream

echo Checking out 'main' branch...
git checkout main

echo Rebasing your 'main' on top of 'upstream/main'...
git rebase upstream/main

echo Done. Your local 'main' is now exactly aligned with upstream.