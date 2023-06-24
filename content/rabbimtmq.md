---
title: "RabbimtMQ"
date: 2023-06-24T13:16:44+02:00
draft: false
categories: ["Data engineering"]
---


## 1. What is RabbitMQ and why do we need it?

RabbitMQ is a message broker, which means that it enables communication between services. Probably the most popular way of communication is TCP, but it has a couple of drawbacks, which rabbitmq solves, for example:
- **lower coupling** between services, so when consumer is down, the messages are not lost
- lots of messages at once don't overwhelm the consumer: it can process them one by one at its own pace

and also provides additional features:
- better **scalability**: there can be more than one consumer with the same amount of effort from producer's perspective (in TCP's case producer sends as many messages as there are receivers, while in the case of RabbitMQ - only one)

In order to provide these functionalities RabbitMQ implements **AMQP** (Advanced Message Queueing Protocol) in which the producer does not produce directly to a message queue, but to an **exchange**, which further redirects the messages to queues. How does an exchange know which queues to send messages to? The user defines **bindings**, which provide this information, and each of the bindings has its own id called **binding key**.


The message can be distrubuted to queues in a couple of ways:
- fanout - exchange distributes messages to all of the binded queues
- direct - message contains a **routing key**, which corresponds to the name of *binding key* of where the message should be sent
- topic - message contains a *routing key* just as in the case of *direct* exchange, but the matching between the routing key and the binding key name does not have to be exact (more info [here](https://www.rabbitmq.com/tutorials/tutorial-five-python.html))
- default (implemented in RabbitMQ, but not part of AMQP) - routing key corresponds directly to the name of the queue, so it bypasses the binding key


Other remarks:
- from the management perspective, RabbitMQ's architecture sets **clear boundaries between DevOps and developers**, who still benefit from tremendous flexibililty of the system, because the whole route of the message is defined in its metadata
- it is **cross-language**, which means that producer and consumer can be written in different programming languages
- provides **acknowledgements**, which are confirmations sent back from the consumer to the queue that the message was received


## 2. Example usage

Let's set RabbitMQ up inside of a docker container on local machine:

```
docker run -d --hostname my-rabbit --name some-rabbit rabbitmq:3
```

Now you are able to access the web UI at [http://localhost:15672/](http://localhost:15672/), and log in with username: guest, password: guest.

In most of my blog posts I describe example usage here, but RabbitMQ already wrote an excellent tutorial, so I would not write a better one anyway. Here's the [link](https://www.rabbitmq.com/tutorials/tutorial-one-python.html).

And here are snippets copy-pasted from the this tutorial, with a minor refactor:

*producer:*

```
import pika

parameters = pika.ConnectionParameters('localhost')
with pika.BlockingConnection(parameters) as connection:
    channel = connection.channel()

    channel.queue_declare(queue='task_queue', durable=True)

    channel.basic_publish(exchange='', routing_key='task_queue', body='Hello World!')
    print(" [x] Sent 'Hello World!'")

```

*consumer:*

```
import pika
import time


def callback(ch, method, properties, body):
    print(" [x] Received %r" % body.decode())
    time.sleep(body.count(b'.'))
    print(" [x] Done")
    ch.basic_ack(delivery_tag=method.delivery_tag)


parameters = pika.ConnectionParameters('localhost')
with pika.BlockingConnection(parameters) as connection:
    channel = connection.channel()

    channel.queue_declare(queue='task_queue', durable=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')


    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='task_queue', on_message_callback=callback)

    channel.start_consuming()
```

## 3. Resources:

- [excellent introductory tutorial prepared by IBM](https://www.youtube.com/watch?v=7rkeORD4jSw)

- [official docs which show how to use RabbitMQ with Python client](https://www.rabbitmq.com/tutorials/tutorial-one-python.html)

- [setup with docker](https://www.rabbitmq.com/download.html)
