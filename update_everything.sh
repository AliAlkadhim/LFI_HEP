#!/bin/bash
jupyter-book clean JupyterBook/
jupyter-book build JupyterBook/
ghp-import -n -o -p -f JupyterBook/_build/html/
git pull
git add .
git commit -m $1
git push

