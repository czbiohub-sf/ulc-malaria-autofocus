#! /usr/bin/env bash

echo -e "\033[1mmypy...\033[0m"
mypy autofocus
echo -e "\033[1mruff...\033[0m"
ruff check . --fix
echo -e "\033[1mblack...\033[0m"
black autofocus tests
