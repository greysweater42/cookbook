#!/usr/bin/env bash

set -e

mod=$1
bs=$2

dev='baseURL = ""'
prod='baseURL = "http:\/\/greysweater42.github\.io"'


if [ "$mod" == "dev" ]
then
    if [ "$bs" == "build" ]
    then
        sed -i "1s/.*/$dev/" config.toml
        sed -i "s|engine\.path\ =\ '\/usr\/bin\/python3'|engine\.path\ =\ '$(pwd)\/venv\/bin\/python'|g" content/*.Rmd
        Rscript -e "blogdown::build_site()"
        sed -i "s|engine\.path\ =\ '$(pwd)\/venv\/bin\/python'|engine\.path\ =\ '\/usr\/bin\/python3'|g" content/*.Rmd
        sed -i "1s/.*/$prod/" config.toml
    elif [ "$bs" == "serve" ]
    then
        sed -i "s|engine\.path\ =\ '\/usr\/bin\/python3'|engine\.path\ =\ '$(pwd)\/venv\/bin\/python'|g" content/*.Rmd
        Rscript -e "blogdown::serve_site()"
        sed -i "s|engine\.path\ =\ '$(pwd)\/venv\/bin\/python'|engine\.path\ =\ '\/usr\/bin\/python3'|g" content/*.Rmd
    else
        echo "You must choose between build and serve"
    fi
elif [ "$mod" == "prod" ]
then
    if [ "$bs" == "build" ]
    then
        sed -i "s|engine\.path\ =\ '$(pwd)\/venv\/bin\/python'|engine\.path\ =\ '\/usr\/bin\/python3'|g" content/*.Rmd
        Rscript -e "blogdown::build_site()"
        sed -i "s|engine\.path\ =\ '$(pwd)\/venv\/bin\/python'|engine\.path\ =\ '\/usr\/bin\/python3'|g" content/*.Rmd
    elif [ "$bs" == "serve" ]
    then
        Rscript -e "blogdown::serve_site()"
    else
        echo "You must choose between build and serve"
    fi
else
    echo "First argument:   [dev|prod]"
    echo "Second argument:  [build|serve]"
fi
