@echo off
echo Fetching 'main' from upstream...
git fetch upstream

echo Checking out 'main' branch...
git checkout main

echo Rebasing your 'main' on top of 'upstream/main'...
git rebase upstream/main

echo Pushing updates on github...
git push --force

echo DONE