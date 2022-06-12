---
title: "Kubernetes"
date: 2022-05-14T13:39:27+02:00
draft: false
categories: ["DevOps"]
---


## What is Kubernetes?

Kubernetes (sometimes "abbreviated" to k8s) is a cluster management tool. A cluster is obviously a group of servers. Managing in this case relies on deployment (in containers), assigning and setting limits on resources like CPU and RAM, rolling back etc.

Kubernetes is arguably the most popular infrastructure-management tool nowadays, which complies to the [infractructure as code](https://en.wikipedia.org/wiki/Infrastructure_as_code) standard.


## How to learn kubernetes?

The easiest way (IMHO) is by setiing up the simplest possible cluster locally on your own laptop. This cluster would consist of one node only (your laptop), but you would be able to run the vast majority of commands (deployment and serving) of your services and get familiar with most of the abstarctions and tools provided by kubernetes.

The tool without which using kubernetes would be truly cumbersome is its console client: `kubectl`.

## Installing minikube and kubectl

Minikube is a mini-version of kubernetes, which provides an extremely simple way to set up a single-node cluster on your local machine. It is used for learning and testing, and we fit well for this purpose. You can install it by following the [installation steps](https://minikube.sigs.k8s.io/docs/start/). Don't forget to start the cluster with `minikube start`!

Installing `kubectl`, i.e. our client to kubernetes is just as easy. Follow [these steps](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/).

>Instead of minikube you can also use [kind](https://kind.sigs.k8s.io/docs/user/quick-start/), which will serve as just as well as `minikube`, but I found the installation process slightly less convenient due to minor troubles during the installation of `go`. The main difference between these two tools is that `minikub` uses virtual machines, but kind uses docker images to simulate a cluster of machines.

## Hello world in kubernetes

The are various use cases for k8s, but one of the popular ones is a REST api, like this one:

`main.py`

```{python}
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World!\n"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
```

which I described in more detail [here](https://greysweater42.github.io/flask/). It creates a simple endpoint on localhost:5000.

```{bash}
$ curl 127.0.0.1:5000
Hello, World!
```

Since kubernetes uses docker images, we'll encapsulate this app into a docker container:

`Dockerfile.yml`
```
FROM python:3.8

RUN pip install flask==2.1.2

WORKDIR /app/
COPY main.py /app/

CMD ["python", "main.py"]

```

which we can then build with
```{bash}
docker build -t api-k8s:latest .
```

and run with

```{bash}
docker run -p 5000:5000 api-k8s:latest
```

Apps are usually set up in a form of `deployment`s. They create `pods`, which are usually single docker containers (can be also groups of containers) that are further run on any of the `nodes` (machines). Kubernetes decides on which node a pod will run, depending on availability of resources (CPU, RAM) on nodes.

Another widely used kubernetes tool is `service`, which provides a stable network address for application, e.g. for a deployment. Keep in mind that pods may die and be restarted automatically by kubernetes, even on a different node. A stable address would be crucial in this case.


Example deployment:

`deployment.yaml`

```
apiVersion: apps/v1
kind: Deployment
metadata:
    name: api-k8s
    labels:
        app: api-k8s
spec:
    replicas: 1
    selector:
        matchLabels:
            app: api-k8s
    template:
        metadata:
            labels:
                app: api-k8s
        spec:
            containers:
                - name: api-k8s
                  image: api-k8s:latest
                  imagePullPolicy: Never
```

Example service:

`service.yaml`

```
apiVersion: v1
kind: Service
metadata:
  name: api-k8s
  labels:
    app: api-k8s
spec:
  ports:
    - protocol: TCP
      port: 5000
      targetPort: 5000
  selector:
    app: api-k8s
```

You run both of them with
```
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Now you can check if you api works. There are at least 2 ways to do that:

- execute a command directly inside a pod; if the command is a request to the API, it should return "Hello world". You can get the pod name with `kubectl get pods`

```
k exec <pod name> -- curl 127.0.0.1:5000
``` 

returns 

```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100    14  100    14    0     0  14000      0 --:--:-- --:--:-- --:--:-- 14000
Hello, World!
```

- port-forward the service to your local host with

```
kubectl port-forward svc/api-k8s 5000:5000
```

and them, from your local machine

```
curl 127.0.0.1:5000
```

returns
```
Hello, World!
```

## Resources

The best resource I've come across so far is [interactive tutorial at kube.academy](https://kube.academy/courses/hands-on-with-kubernetes-and-containers/lessons/the-top-takeaways-from-the-course).
