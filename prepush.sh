printf "\nruff check..."
ruff check autofocus/ --ignore E501 --ignore E722 --fix

printf "ruff formatting..."
ruff format autofocus/
