---
title: "bash"
date: 2020-01-19T14:31:25+01:00
draft: false
categories: ["DevOps"]
---

## 1. What is bash and why would you use it?

- bash is a Unix shell and commmand language, which makes performing any task you want on Unix/Linux system programatically

- anything you've ever done using OS GUI can be done with bash. The main benefit is the possibility to automate any task and keep track of its changes with git

## 2. Why am I writing this tutorial?

- bash is the most basic Linux

> disclaimer: all the GNU/Linux/Unix systems I call Linux. I think it became a standard now. I know it's incorrect (e.g. OSX), but shorter and everybody seems to understand ;)

skill but it is quite oldschool and sometimes daunting, especially because it probably is not you main programming language, as a developer. Even the simplest commands (if statements, assigning variables, executing commands, for loops) have rather non-intuitive, quirky syntax. This tutorial will hopefully serve me as a reference during future development.

## 3. Examples

#### shebang

Two possible ways:
```{bash}
#!/bin/bash
#!/usr/bin/env bash
```
The second one is more popular.

#### running code

There are at least three possible ways to execute a bash script:

- `bash script.sh` - creates a new bash session and runs the code in it. The variables in your curretn session will be available, however if you `exit` your code, the current (~host) session will not exit. If you assign a variable using `export`, it will not be available further on.

- `source script.sh` and ` .script.sh` - so far I haven't noticed any differences. If there are differences, sorry for my ignorance.

#### variables assignment

- simple interior assignment - these variables are available only in the script
```{bash}
x=10
var=hi
```

- exterior assignment - these variables will be available after the script was run, unless it was run with `bash script.sh` command:
```{bash}
export var=hello
```

- assigment of the result of executed statement - if you want to execute a statement in shell and assign its result to the variable, use the following sytnax:
```
val=$(date)
var=$(ls | wc -l)
```

#### if statement

Keep in mind that you *have to* leave spaces between a condition inside square brackets and the brackets themselves:
```{bash}
if [ hi == hi ]
then
    echo yes
elif [ hi == hello ]
then
    echo no
else
    echo what
fi
```

A common practice is to write `if [ hi == hi ]; then` in one line.

Logical operators:

- not `[ ! 1 == 1 ]`

- and `[ cond1 && cond2 ]`

- or `[ cond1 || cond2 ]`

- is equal - for strings it's simply `==`, while for integers it's `[[ $var -eq 1 ]]`. String comparison should be sufficient for most cases, unless you explicitly declare integers.

- combining `[[ cond1 ]] && [[ cond2 || cond3 ]]`

Check out [this link](https://stackoverflow.com/a/6270803/10930429) for more detailed information.

#### for loop

There are many objects you can iterate over. You will find a pretty exhaustive list [here](https://www.cyberciti.biz/faq/bash-for-loop/) with examples of `break` and `continue`. And here's a trivial example:
```
for i in 1 2 3 4 5
do
   echo $i
done
```

##  4. Links

- [excellent bash cheatsheet](https://devhints.io/bash)
