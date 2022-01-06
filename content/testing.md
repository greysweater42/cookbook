---
title: "testing"
date: 2018-02-04T12:02:23+01:00
draft: false
categories: ["R", "Python"]
---

## 1. What is testing and why would you do it? 

* testing or [test-driven development](https://en.wikipedia.org/wiki/Test-driven_development) (TDD) is a discipline, which relies on writing a test for every functionality *before* creating it (in reality you often end up writing tests afterwards)

* at first the test will fail, as we have not provided the proper functionality yet. Our goal is to fulfill this functionality, so the test will pass (in reality you usually write tests so that later you can code more... inattentively, and tests point out your mistakes before reaching pruduction environment)

## 2. "Hello World" examples 

### R (testthat) 

Let's go through testing two simple functions:

```
library(testthat)

context("simple example tests")

divider <- function(a, b) {
  if (b == 0) {
    warning("One does not simply divide by zero.")
    return(NULL)
  }
  x <- a / b
  return(x)
}

test_that(
  desc = "check proper division by `divider()`",
  code = {
    expect_equal(divider(10, 5), 2)

    expect_warning(divider(2, 0))

    expect_null(suppressWarnings(divider(10, 0)))
  }
)

summer <- function(a, b) {
  x <- a + b
  if (x == 5) x <- x + 1  ## a strange bug
  return(x)
}

test_that(
  desc = "check proper summing by `summer`",
  code = {
    expect_equal(summer(2, 2), 4, info = "two and two is four")
    expect_equal(summer(2, 3), 5, info = "two and three is five")
  }
)
```

What have we done here?

* we loaded `testthat` package;

* we provided a context - this is the first message that appears in tests summary and serves as a title for this particular group of tests;

*Couriously, in order to run test properly, [you *have to* provide context](https://stackoverflow.com/questions/50083521/error-in-xmethod-attempt-to-apply-non-function-in-testthat-test-when)*.

* we wrote a function `divider`, which divides two numbers and `summer`, which adds two numbers (clever!); as you can see, there is a strange bug in `summer`

* `test_that` functions belong to `testthat` package and they will check if these functions run properly;

* there are various types of `expect_...`, a pretty interesting one is `expect_fail(expect_...())`;

* you should provide additional description (`desc`) to each `test_that` function, so you could easily find which test failed; you can also provide info for every single `expect_...`.


Now we can run our tests with `testthat::test_file(<name_of_file>)` or `testthat::test_dir('tests')`, depending on where you store your functions with tests. In production, you obviously keep testing functions in separate files, preferably in a `tests` folder and each file is called `test_...`.  In that case you source all the functions you want to test simply with `source()`.



Testing our above file will result in:

```
R> test_file('test.R')                                                          
✔ | OK F W S | Context
✖ |  4 1     | simple example tests
────────────────────────────────────────────────────────────────────────────────
test.R:37: failure: check proper summing by `summer`
summer(2, 3) not equal to 5.
1/1 mismatches
[1] 6 - 5 == 1
two and three is five
────────────────────────────────────────────────────────────────────────────────

══ Results ═════════════════════════════════════════════════════════════════════
OK:       4
Failed:   1
Warnings: 0
Skipped:  0
```

As we can see, 4 tests have passed, one has failed. The one that failed was in 37th line of the test file, when we were 'checking proper summing by summer'. According to `summer`, two and three is not five `¯\_(ツ)_/¯`.

### Python (pytest) 

When testing Python code, I usually use `pytest`, however `unittest` still seems seems to be standard among the community, quite surprisingly. The main advantage of `pytest` comparing to `unittest` is it's simplicity: it may take even less than a minute to start being productiove with testing. 

Let's create a very simple function and save it to a file:

func.py
```
def divider(a, b):
    return a / b
```

and a functioh which will test it's validity:

test_divider.py
```
from func import divider


def test_divider():
    assert divider(10, 2) == 5
```

*A testing file's name should begin with `test_`, so pytest would recognize it. If you create many of these files, you can keep them in one file called `tests`.*

Now run `pytest` command from console and you will see
```
$ pytest
================================ test session starts ================================
platform linux -- Python 3.5.2, pytest-3.7.2, py-1.5.4, pluggy-0.7.1
rootdir: /home/tomek/cookbook/scratchpad/testing, inifile:
plugins: pylama-7.4.1
collected 1 item                                                                    

test_divider.py .                                                             [100%]

============================= 1 passed in 0.01 seconds ==============================
```

that the test passed. If you want to receive more specific information, use the verbose flag: `pytest -v`.

What happens if the test fails? Let's write another test, which will clearly cause an error:

test_divider.py
```
from func import divider


def test_divider():
    assert divider(10, 2) == 5

def test_divider_by_zero:
    assert divider(10, 0)
```

Running pytest causes:
```
$ pytest
=================================== test session starts ===================================
platform linux -- Python 3.5.2, pytest-3.7.2, py-1.5.4, pluggy-0.7.1
rootdir: /home/tomek/cookbook/scratchpad/testing, inifile:
plugins: pylama-7.4.1
collected 2 items                                                                         

test_divider.py .F                                                                  [100%]

======================================== FAILURES =========================================
__________________________________ test_divider_by_zero ___________________________________

    def test_divider_by_zero():
>       assert divider(5, 0)

test_divider.py:9: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

a = 5, b = 0

    def divider(a, b):
>       return a / b
E       ZeroDivisionError: division by zero

func.py:2: ZeroDivisionError
=========================== 1 failed, 1 passed in 0.02 seconds ============================
```

As we can see, dividing by zero raises a ZeroDivisionError.

### Python testing extensions

- **fixtures** are used to extract a common, integral and independent part of many test functions to a separate object. This can be e.g. a database connection or a csv file or, in fact, any object.

- **mocking** - its name is quite self-explanatory: it substitutes a function/method/object from e.g. a separate package with the one provided by us. May be used for mocking database connections, http requests, reading csv files - virtually any I/O operation, or if we test a method, which uses other methods of a class: we can mock the other methods, as they are tested separately. But beware! Never change the behavior and tests of the other methods! Actually you should never change the tests anyway.

## 3. Useful links 

* R:

    * [Hadley Wichkam's article on testthat](https://journal.r-project.org/archive/2011/RJ-2011-002/RJ-2011-002.pdf)

    * [usethis](https://github.com/r-lib/usethis) - useful if you want to test a package

* Python:

    * [Test driven development with Python](https://learning.oreilly.com/library/view/test-driven-development-with/9781449365141/)

    * [Probably the best book with snippets and "Hello World" examples: The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/tests/)

    * [last but not least - excellent Pytest's docs](https://docs.pytest.org/en/6.2.x/contents.html)

## 4. Subjects still to cover in this blogpost

* unittest, coverage (TODO)
