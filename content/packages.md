---
title: "packages"
date: 2018-02-04T12:06:07+01:00
draft: false
categories: ["R"]
image: "packages.jpg"
tags: ["R", "DevOps"]
---

## 1. What are R packages and why would you use them?

* R packaging is a convenient way to store and share your R code. 

* It lets you incorporate testing with [testthat](http://tomis9.com/testthat) specially prepared tools (you can use testthat without creating a package, but it's slightly more complicated).

* It lets you easily list dependencies with [packrat](http://tomis9.com/packrat). You can also achieve this without using a package.

* You can easily version your code. Yes, you can use github tags.

You may say: well, I can benefit from all of these advantages by simply keeping my code in a git repo and you know what? You would be right. Maybe except the dependencies, however if you do explicitly tell which function comes from which foreign package, users will be fine.

In general, I don't recommend encapsulating your code into a package, unless you want to publish it on cran. Your workflow will require one more step whenever you change your code, i.e. you will have to reinstall the package.

## 2. Creating and installing a package

A `devtools` package makes creating and developing packages easy:
```
library(devtools)
```

Set the directory where you want your package to live:
```
setwd('~/')
```

Now create the package with `devtools::create()`. 

```
devtools::create('helloWorldPackage')
setwd('./hello_world_package')
```
*You can of course skip the `devtools::` part once you have loaded the `devtools` package using `library(devtools)`. But keep in mind that when you write a package and whenever you use any function from a foreign package, always use `foreign_package_name::` naming convention. Otherwise if a user has loaded a function with the same name, but imported from a different package, the other function will be run.*

## 3. Folders and files in your package directory

`Devtools` creates a frame for your package, which consists of:

```
.

├── DESCRIPTION
├── helloWorldPackage.Rproj
├── NAMESPACE
└── R
1 directory, 3 files 
```

`R` is a directory where you will store your R functions.

`DESCRIPTION` is the file which you should update with the basic information about your package. It already contains default values, so don't worry if you forget doing this. So many people forget. What a nightmare. Please don't forget about updating this file.

Default `DESCRIPTION` file:
```
Package: helloWorldPackage
Title: What the Package Does (one line, title case)
Version: 0.0.0.9000
Authors@R: person("First", "Last", email = "first.last@example.com", role = c("aut", "cre"))
Description: What the package does (one paragraph).
Depends: R (>= 3.4.4)
License: What license is it under?
Encoding: UTF-8
LazyData: true
```

`NAMESPACE` contains a list of functions which you want to be exported from the package, i.e. visible to the users. You don't have to update it by hand, as `devtools::document()` function:

```
devtools::document()  # creates manual/documentation using function from roxygen2 package
```

does it automatically. 

Last but not least, once you've finished writing your code:

```
devtools::install('.')  # install package
```

install the package. Since the you will be able to load the functions from this package using `library(helloWorldPackage)`.

## 3. Documenting your package

As I mentioned in the previous point, you don't have to build documentation by yourself, but you have to provide all the important information, so it could be built automatically. Every function that you want to export should begin with something similar to python's docstring:

```
#' Markov clustering function
#'
#' Markov clustering function algorithm based on Data Mining and Analysis: 
#  Fundamental Concepts and Algorithms, Zaki, Meira, p.462
#' @param M adjacency matrix
#' @param inflation inflation parameter
#' @param expansion expansion parameter
#' @param eStop stop criterium - Frobenius norm of matrix M_t - M_{t-1} 
#' @param max_it maximum number of iterations
#' @keywords markov clustering, graph clustering
#' @import expm
#' @export
#' @examples 
#' A <- rbind(c(1, 1, 0, 1, 0, 1, 0),
#'            c(1, 1, 1, 1, 0, 0, 0),
#'            c(0, 1, 1, 1, 0, 0, 1),
#'            c(1, 1, 1, 1, 1, 0, 0),
#'            c(0, 0, 0, 1, 1, 1, 1),
#'            c(1, 0, 0, 0, 1, 1, 1),
#'            c(0, 0, 1, 0, 1, 1, 1))
#' markovClust(A)
markovClust <- function(M, inflation=2.5, expansion=2, e_stop=0.001, 
                        max_it=100, name) {
    ...
```

It has a well-defined structure.

* *Markov slustering function* - name of a function.

* *Markov clustering function algorithm ...* - description of a function.

* @param - description of a parameter of a function.

* @keywords make your function easy searchable.

* @import - which foreign packages the function uses.

* @export - should the function be visible to end users? If you do not include this line, function will not be exported.

* @examples - example usage of a function.

Under the "docstring" you provide the definition of a function.

A good practice is to keep only one function in each R file, but if the functions are short, they are not exported or depend on each other, you may keep them in one file.

## 3. Useful links

A nice basic tutorial on writing R packages is available [here](https://hilaryparker.com/2014/04/29/writing-an-r-package-from-scratch/).

## 4. Subjects still to cover

* packrat (TODO)
