---
title: "hadoop"
date: 2018-12-04T21:32:35+01:00
draft: false
categories: ["Data engineering"]
---

## 1. What is hadoop and why would you use it? 

* hadoop is the first ever popular big data tool;

* it lets you *quickly* compute huge amounts of data thanks to dividing computation into many machines i.e. a cluster of machines; (*quickly* comparing to a standard, one-machine approach);

* you can store and easily access huge amounts of data thanks to hadoop's distributed file system (hdfs);

* in general, hadoop is a cornerstone of big data.


## 2. Installation 

In production environment Hadoop should be installed on several interconnected machines, called a cluster. I'm not going to explain how to do this, as this is usually a task for sysadmins/DevOps guys, and not for data scientists. However it is very usufeul to have your own one-node hadoop installation on your laptop for practising basic big data solutions. This is what we're going to install in this section.

[Here](hadoop.sh)'s a little script that I wrote which installs and runs hadoop. You can execute it with
```
sudo bash hadoop.sh
```
but I recommend running it line by line, so you would understand what you are doing and what is needed to have hadoop running.


## 3. "Hello World" examples 

You probably expect that knowing Hadoop will let you do Big Data. It would, but in reality nobody uses Hadoop to run data processing anymore. [Spark](http://tomis9.com/spark) and [Hive](http://tomis9.com/hive) have much nicer apis, and do the same tasks faster.

But there is still one functionality that was not replaced: a file system, called *hadoop distributed file system* or *hdfs*, where limited disk capacity is no longer a problem.

There are a couple of useful commands to remember. Their resemblance to bash commands is quite obvious.

ls - list files in a given directory
```
hdfs dfs -ls <dir>
```

mkdir - create a directory
```
hdfs dfs -mkdir <path>
```

put - upload file to hdfs
```
hdfs dfs -put <localSrc> <dest>
```

copyFromLocal - copy file local to hdfs, obviously
```
hdfs dfs -copyFromLocal <localSrc> <dest>
```

get - download file from hdfs
```
hdfs dfs -get <src> <localDest>
```

copyToLocal - copy file from hdfs to local, obviously
```
hdfs dfs -copyToLocal <src> <localDest>
```

mv - move file from one directory to another or rename a file
```
hdfs dfs -mv <src> <dest>
```

cp - copy file from one directory to another
```
hdfs dfs -cp <src> <dest>
```

rm - remove a file from hdfs (`-rm -r` - remove recursively)
```
hdfs dfs -rm <dir>
```

More commands are available [here](https://data-flair.training/blogs/top-hadoop-hdfs-commands-tutorial/).

## 4. Subjects still to cover 

* Hadoop GUI (TODO)

* example of MapReduce: wordcount (TODO)

* [RHadoop](https://github.com/RevolutionAnalytics/RHadoop/wiki) (TODO)
