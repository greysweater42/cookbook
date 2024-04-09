---
title: "docker-compose biolerplate"
date: 2024-04-09T09:41:56+02:00
draft: false
categories: ["DevOps"]
---


## docker-compose boilerplate

I don't write docker-compose files that often. Usually there is only one docker-compose file per project, and when each project lasts for a couple of months... I tend to forget what the components of a docker-compose.yml file are.

Here's a boilerplate for a `docker-compose.yml` file:

```
version: '3'
services:

  app:
    build: .
    image: test_simple_http_server:latest
    command: ["python3", "-m", "http.server", "9000"]
    environment:
      - SOME_ENV_VARIABLE=test
    volumes:
      - .:/app
    ports:
      - "9000:9000"
    depends_on:
      - mysql

  # https://hub.docker.com/_/mysql
  mysql:
    image: mysql
    command: --default-authentication-plugin=mysql_native_password
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: example
    ports:
      - "3306:3306"
```

for a simple `Dockerfile`:

```
from python3.10
```