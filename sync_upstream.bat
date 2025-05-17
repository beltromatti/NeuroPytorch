@echo off
echo Fetching 'main' from upstream...
git fetch upstream main:upstream-main

echo Checking out 'main' branch...
git checkout main

echo Merging 'upstream-main' into 'main'...
git merge upstream-main

echo Done. Your local 'main' branch is now up to date.