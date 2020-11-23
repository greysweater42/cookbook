---
title: "Parallel"
date: 2020-10-15T11:12:24+02:00
draft: true
categories: ["scratchpad"]
---

```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```
