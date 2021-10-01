---
title: "async/threads"
date: 2021-09-29T00:14:48+02:00
draft: false
categories: ["Python", "Data engineering"]
---


## 1. What does "asynchronous" and "threading" mean and why is it important?

Let's begin with the definition of processes and threads provided [Gerald of Stackoverflow](https://stackoverflow.com/a/200511):

>A process is a collection of code, memory, data and other resources. A thread is a sequence of code that is executed within the scope of the process. You can (usually) have multiple threads executing concurrently within the same process.

*Asynchrounous* usually refers to using many processes, while *threading* - many threads within one process.

* It is particurarily useful in data engineering, because it can significantly reduce the time of data processing (which can also be done if you use [PyPy](https://www.pypy.org/))

* Unfortunately there are at leat 4 Python packages used for async/threading, which makes working with them a little confusing. What are the differences between them and which one to choose?

## 2. Traditional approaches

### 2.1 concurrent.futures

This chapter was inspired by an excellent book [Fluent Python by Luciano Ramalho](https://www.oreilly.com/library/view/fluent-python/9781491946237/), chapter 17: Concurrency with Futures.

#### 2.1.1 threads

Let's say we want to run a trivial function `sleep_int` several times. Obviously we want to do this concurrently / in parallel.

```{python}
import time
from concurrent import futures
from datetime import datetime

from tqdm import tqdm  # pip install tqdm


def sleep_int(i):
    time.sleep(i)
    return i
```

This is how we can approach this using threads. In general, as threads are all in the same process, GIL (Global Interpreter Lock) would not allow our `sleep_int` runs to run at the same time, unless they are `sleep`ing or read/write data (input/output). In our case we `sleep`, so we can use threads in the same way as processes.

```{python}
with futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(sleep_int, range(5))
    print(type(results))
    for result in results:
        # time when a task is finished and its result
        print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {result}")
```
```
<class 'generator'>
17:57:44.577651: 0
17:57:45.578580: 1
17:57:46.579840: 2
17:57:47.580866: 3
17:57:49.582768: 4
```

We can see that `results` is of class `generator`, which means in practice, that:
- it is an iterable object, which resembles lists,
- but we do not known its length until it tells us "this is the end",
- and items of generator are *ephemeral*, which means they disappear as soon as you move to another item of the generator (I guess *ephemeral* may not be the best word to describe this feature, but you get the idea).

Another interesting fact is that we defined 3 threads (`max_workers=3`) to be run simultaneously, so we can see that:
- runs 0, 1 and 2 started at the same time (around `17:57:44.577`) and they occupied all 3 available threads,
- but run 3 started also around `17:57:44.577`, because run 0 took 0 seconds,
- and run 4 started around `17:57:45.58`, right after run 1 ended and freed one of the threads.

#### 2.1.2 processes

There are only two difference between using processes and threads in `concurrent.futures`:
- instead of `ThreadPoolExecutor` we write `ProcessPoolExecutor`,
- and our function, in this case `sleep_int` uses separate processes, not just separate threads of one process, so it can run any group of functions simultaneously, not only those which `sleep` or read/write (input/output).

An analogical example of using multiple processes:
```
with futures.ProcessPoolExecutor(3) as executor:
    results = executor.map(sleep_int, range(5))
    for result in results:
        print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {result}")
```

#### 2.1.3 progress bars and error handling
We usually want to run our code in parallel/concurrently to save time, when the task is time-consuming. But solving this kind of problem has two disadvantages:
- we do not know how long the task is going to last
- we fear error on one of the runs, which forces us to run everything again.

Luckily there are ways to address these issues:
- use progress bar, in this case tqdm
- and handle errors, preferably outside of sleep_int function

Unfortunately we can not use our convenient `executor.map` friend this time.
We need to dive a little deeper into `concurrent.futures` and use `futures.as_completed()`, which takes an interable object of futures to be executed and generates their results when they end.

```
n = 5
with futures.ProcessPoolExecutor(max_workers=3) as executor:
    to_do = [executor.submit(sleep_int, i) for i in range(n)]
    done_iter = futures.as_completed(to_do)
    # since done_iter is a generator, we should provide total length to tqdm
    for future in tqdm(done_iter, total=n):
        try:
            res = future.result()
        except ValueError: ## e.g. "sleep length must be non-negative"
            res = 0    
        print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {res}")
```

### 2.2 threading

Threading in Python may be useful only for input/output bound tasks, but it works also work sleep function, as input/output and sleep both can be run simultaneously on multiple threads, despite GIL.
Anyway the following example is not particurarily useful, because `threading` is a more low-level package comparing to `concurrent`, so in practice it is more convenient to use `concurrent`.

To sum up, in practice if you really want to use threads, you should use `concurrent.futures`, but in general you will want to use many processes, not many threads of one process.

```{python}
import threading
import time
from datetime import datetime


def sleep_int(i):
    time.sleep(i)
    print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
    return 1

threads = []

for i in range(5):
    t = threading.Thread(target=sleep_int, args=[i])
    t.start()
    threads.append(t)

for thread in threads:
    thread.join()

print("done")
```
```
00:37:10.190047: 0
00:37:11.191345: 1
00:37:12.192518: 2
00:37:13.193746: 3
00:37:14.194839: 4
done
```

As you can see, for each run of `sleep_int` function we created a separate thread using `threading.Thread` constructor, started them - this is the moment when the computation starts in the background, on a separate thread - and added each of them to the `threads` list, so we could `.join()` each of the threads later. Joining threads means that the main thread in which the script is run will wait until all the `threading.Thread`s are over before continuing, in this case "done" was printed when all the `threading.Thread`s were finished.


### 2.3 multiprocessing

#### 2.3.1 simplest case

All you ever need to know about multiprocessing is available in [Python's docs on multiprocessing](https://docs.python.org/3/library/multiprocessing.html), but as this is quite a demanding and time-consuming lecture, let me shorten it a little.

`multiprocessing` at first sight works exactly the same as `concurrent.futures.ProcessPoolExecutor`, as you can see in the example below:

```
import time
from datetime import datetime
from multiprocessing import Pool


def sleep_int(i):
    time.sleep(i)
    return i


with Pool(3) as executor:
    print(datetime.now().strftime('%H:%M:%S.%f'))
    results = executor.map(sleep_int, range(5))
    print(type(results))
    for result in results:
        print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {result}")
```
```
22:52:57.521025
<class 'list'>
22:53:02.528105: 0
22:53:02.528179: 1
22:53:02.528197: 2
22:53:02.528210: 3
22:53:02.528222: 4
```
but there is one quite important difference: the `results` object is not a generator, but a list, so we may receive the results as soon as all the futures are ended. In this case, after 5 seconds (`sleep_int(4)` started right after `sleep_int(1)` finished, just as a process was freed). A list however has an advatage: it's objects are always available, i.e. they are not *ephemeral*.

#### 2.3.2 using tqdm with multiprocessing

There is a simple way for executor to return a generator instead of a list: an `imap` method instead of `map`. This minor change lets us use tqdm for monitoring progress, but we still have to provide the length of the generator by ourselves.

```{python}
from tqdm import tqdm

with Pool(3) as executor:
    results = executor.imap(sleep_int, range(5))
    for result in tqdm(results, total=5):
        print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {result}")
```

#### 2.3.3 similarity to threading

Multiprocessing API is very similar to threading's. If you don't use `Pool`, you define separate processes in a loop, start them and after that, join them, but this is rarely used in practice.

```{python}
processes = []

for i in range(5):
    p = Process(target=sleep_int, args=[i])
    p.start()
    processes.append(p)

for process in processes:
    process.join()
```

#### 2.3.4 global variables

In general when you use multiprocessing, separate processes do not communicate with each other nor they do not have any side effects. But there are two posiibilites if you really need one of these features:

- shared memory

- server process

both of them are fairly straightfoward and are thoroughly described in [docs](https://docs.python.org/3/library/multiprocessing.html#sharing-state-between-processes).

## 3. More sophisticated approaches

### 3.1 asyncio

#### 3.1.1 coroutine - high-level API

According to [asyncio docs](https://docs.python.org/3/library/asyncio.html),

>Coroutines declared with the async/await syntax is the preferred way of writing asyncio applications

so it seems to be different to most tutorials I've seen so far (e.g. [sentdex's](https://www.youtube.com/watch?v=BI0asZuqFXM) or *Fluent Python*, Chapter 18: Concurrency with asyncio), which propose low-level API: event loops, which I describe later in this blog post.

BTW, *Fluent Python* suggests using an older syntax for `asyncio`, specifically [generator-base coroutines](https://docs.python.org/3/library/asyncio-task.html#generator-based-coroutines), which currently is deprecated.

To be clear, in this context a *coroutine* is a function declared as `async def` function.


A simple example of asyncio using coroutine:

```{python}
import asyncio
from datetime import datetime


async def sleep_int(i):
    await asyncio.sleep(i)  # asyncio strongly prefers asyncio.sleep over time.sleep
    print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
    return i


async def coro():
    # aws - short for awaitables
    aws = [asyncio.create_task(sleep_int(i)) for i in range(5)]
    return await asyncio.gather(*aws)


results = asyncio.run(coro())  # run coroutine
print(results)
```
```
10:44:33.451405: 0
10:44:34.452606: 1
10:44:35.452879: 2
10:44:36.453250: 3
10:44:37.453631: 4
[0, 1, 2, 3, 4]
```

Once you get used to the somehow unusual syntax (`async` and `await`), concurrency with `asyncio` has rather similar API to `multiprocessing` or `threading`: we define a function, which will be run asynchronously, `create_task`s (called `Process`es in multiprocessing) and `await` for them to finish (called `join` in multiprocessing).

Unfortunately defining the number of available processes (or Pool size in multiprocessing) is more difficult and requires using [semaphores](https://stackoverflow.com/a/48486557).

#### 3.1.2 event loop - low-level API

Here's an example of asyncio with event loop instead of coroutine. This solution is not recommended by asyncio docs, but it is more popular among asycion tutorials. It works exactly the same as coroutine described before, except you cannot retrieve what your function returns.

```{python}
import asyncio
from datetime import datetime


async def sleep_int(i):
    await asyncio.sleep(i)  # asyncio strongly prefers asyncio.sleep over time.sleep
    print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
    return i


loop = asyncio.get_event_loop()
tasks = [loop.create_task(sleep_int(i)) for i in range(5)]
try:
    loop.run_until_complete(asyncio.wait(tasks))
finally:
    loop.close()
```

The `try-except-finally` block is used here to assure that the event loop will always be closed.

### 3.2 ray

[Ray](https://github.com/ray-project/ray) ([docs](https://docs.ray.io/en/latest/index.html)) is a multitool for machine learning, and one of its features is multiptocessing. The syntax is very simple and similar to other packages:

```
import time
from datetime import datetime

import ray

ray.init()


@ray.remote
def sleep_int(i):
    time.sleep(i)
    print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
    return i


futures = [sleep_int.remote(i) for i in range(5)]
results = ray.get(futures)
print(results)
```
```
(pid=17741) 12:17:50.872821: 0
(pid=17740) 12:17:51.873061: 1
(pid=17738) 12:17:52.869560: 2
(pid=17737) 12:17:53.870272: 3
[0, 1, 2, 3, 4]
(pid=17735) 12:17:54.875371: 4
```

### 3.3 celery

Celery works quite differently comparing to the tools described before:
- you run a separate instance of celery application, called `worker`, on which your `task` will be executed,
- and a separate queue app, usually redis,
- and then you can prompt the worker (via queue) to run the task.

There is an excellent [documentation](https://docs.celeryproject.org/en/stable/index.html) with pretty good [introductory tutorials](https://docs.celeryproject.org/en/stable/getting-started/index.html), but from my own experience the best way to learn how celery works is by trial-and-error: just try out various ideas and see if they work or not, then move on to documentation to understand *why* they work.

Here's a simple example of celery:

```{bash}
pip install celery
pip install sqlalchemy  # for sqlite backend
```

tasks.py
```{python}
from celery import Celery
import time

BROKER_URL = "sqla+sqlite:///celery.db"
BACKEND_URL = "db+sqlite:///celery_results.db"
app = Celery(broker=BROKER_URL, backend=BACKEND_URL)


@app.task
def sleep_int(i):
    time.sleep(i)
    return i
```

We've created a `Celery` app instance and provided information about the queue. In the case above we chose `sqlite`, but in production you would rather use redis in docker conteiner or or on a separate server. Then the URLs would look more like this:
```{python}
BROKER_URL = 'redis://172.18.0.2:6379/0'
BACKEND_URL = 'redis://172.18.0.2:6379/0'
```

Run the celery instance with:
```{bash}
celery -A tasks worker --loglevel=INFO
```
where `tasks` is a name of the file `tasks.py`. This should print something like this:
```
 -------------- celery@td v5.1.2 (sun-harmonics)
--- ***** ----- 
-- ******* ---- Linux-5.10.0-1045-oem-x86_64-with-glibc2.31 2021-10-01 13:47:46
- *** --- * --- 
- ** ---------- [config]
- ** ---------- .> app:         __main__:0x7f8f0bd2f370
- ** ---------- .> transport:   sqla+sqlite:///celery.db
- ** ---------- .> results:     sqlite:///celery_results.db
- *** --- * --- .> concurrency: 8 (prefork)
-- ******* ---- .> task events: OFF (enable -E to monitor tasks in this worker)
--- ***** ----- 
 -------------- [queues]
                .> celery           exchange=celery(direct) key=celery
                

[tasks]
  . tasks.sleep_int

[2021-10-01 13:47:46,667: INFO/MainProcess] Connected to sqla+sqlite:///celery.db
[2021-10-01 13:47:46,681: INFO/MainProcess] celery@td ready.
```
Then you can write a separate file with the following content:

run.py
```{python}
from tasks import sleep_int
for i in range(5):
    sleep_int.apply_async((i,))
```
and run it with `python run.py` (or run the libes above from python console) and see the results in celery log:
```
[2021-10-01 13:48:18,869: INFO/MainProcess] Task tasks.sleep_int[205c72bb-60ca-4166-a7a1-53dbb05f952a] received
[2021-10-01 13:48:19,920: INFO/ForkPoolWorker-1] Task tasks.sleep_int[205c72bb-60ca-4166-a7a1-53dbb05f952a] succeeded in 1.0483712660006859s: 1
[2021-10-01 13:48:46,032: INFO/MainProcess] Task tasks.sleep_int[2aa3c291-3807-492f-b8bc-3fb3ed282759] received
[2021-10-01 13:48:47,066: INFO/ForkPoolWorker-2] Task tasks.sleep_int[2aa3c291-3807-492f-b8bc-3fb3ed282759] succeeded in 1.0317802720001055s: 1
[2021-10-01 13:52:10,168: INFO/MainProcess] Task tasks.sleep_int[edfdb82a-1c37-4004-b87e-6195646f8f57] received
[2021-10-01 13:52:10,181: INFO/MainProcess] Task tasks.sleep_int[3131e2bd-8454-4828-b6ec-7c428b2d2e06] received
[2021-10-01 13:52:10,192: INFO/MainProcess] Task tasks.sleep_int[4de05146-0a48-4980-88fb-414a37c74514] received
[2021-10-01 13:52:10,202: INFO/ForkPoolWorker-3] Task tasks.sleep_int[edfdb82a-1c37-4004-b87e-6195646f8f57] succeeded in 0.03295764599897666s: 0
[2021-10-01 13:52:10,206: INFO/MainProcess] Task tasks.sleep_int[c4bb963f-72ef-44c9-bd97-4cb3f432caaa] received
[2021-10-01 13:52:10,216: INFO/MainProcess] Task tasks.sleep_int[4b9db51e-13f1-427d-9f15-83e3cca0f10a] received
[2021-10-01 13:52:11,214: INFO/ForkPoolWorker-4] Task tasks.sleep_int[3131e2bd-8454-4828-b6ec-7c428b2d2e06] succeeded in 1.03144066400273s: 1
[2021-10-01 13:52:12,225: INFO/ForkPoolWorker-5] Task tasks.sleep_int[4de05146-0a48-4980-88fb-414a37c74514] succeeded in 2.0326271019985143s: 2
[2021-10-01 13:52:13,241: INFO/ForkPoolWorker-6] Task tasks.sleep_int[c4bb963f-72ef-44c9-bd97-4cb3f432caaa] succeeded in 3.0335887920009554s: 3
[2021-10-01 13:52:14,251: INFO/ForkPoolWorker-7] Task tasks.sleep_int[4b9db51e-13f1-427d-9f15-83e3cca0f10a] succeeded in 4.03322789899903s: 4
```

`celery` ran all the tasks and printed their results. Retrieving the results back to the run.py is possible, but rather complicated and this is not really what celery was meant to do. Beside that minor disadvantage, celery has many advantages:
- you can easily scale it into many machines. There may be a separate worker on each machine, which can consume task prompts from queue (redis), which can also be on a separate machine. Hence it works perfectly with Kubernetes
- as it can scale to many machines, it is comparable to spark, but celery is (IMHO) a much mature and easy-to-maintain piece of software
- it has a reasonably good GUI called [flower](https://flower.readthedocs.io/en/latest/)
- you can write your own asynchronous pipelines using e.g. [chains](https://docs.celeryproject.org/en/stable/userguide/canvas.html#chains)

To sum up, using celery is more of an architectural decision, because it usually runs on several machines. I was lucky to be working with this tool for more than a year and I highly recommend it.

### 3.4 (py)spark
 
I wrote a [separate blog post about spark](https://greysweater42.github.io/spark). Keep in mind that spark is used primarily for big data.
