#!/bin/bash

jupyter-book build JupyerBook
ghp-import -n -o -p -f JupyterBook/_build/html/
git add .
git commit -m "$1"
git push

