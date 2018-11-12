---
title: "docker"
date: 2018-11-09T23:01:35+01:00
draft: false
image: "whale.jpg"
tags: ["DevOps"]
---

You can download it from [here](https://www.docker.com/products/docker-engine#/linux) or simply download and install with `sudo apt install docker.io`.

A pretty long, but credible tutorial is available [here](https://docker-curriculum.com/).

By definition, it'a a 

"In simpler words, Docker is a tool that allows developers, sys-admins etc. to easily deploy their applications in a sandbox (called containers) to run on the host operating system i.e. Linux."

Comparing to Python, it's basically a virtualenv, but for the whole OS.
Or this is some sort of a virtual machine.

Check if docker is properly installed:
```
sudo docker run hello-world
```

Let's download an example of docker Image from the Docker Registry:
```
sudo docker pull busybox
```

This command lists all images and their statuses: 
```
sudo docker ps -a
```

In order to run an image, type: 
```
sudo docker run sandbox
```
You may also want to work in an image in an interactive mode:
```
sudo docker run -it sandbox sh
```


View of all the images:
```
sudo docker images
```

Stop a specific image:
```
sudo docker stop image_id
```


Well, there is actually many various topics on docker, so maybe you should just go through the tutorial.
It's rather straightforward.

One third of this tutorial should be enough :)


Plus - how to use it on [gitlab](https://gitlab.iiit.pl/help/user/project/container_registry).


### A good [book](http://pepa.holla.cz/wp-content/uploads/2016/10/Using-Docker.pdf)

docker volumes
