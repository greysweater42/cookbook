---
title: "airflow"
date: 2018-08-14T11:51:12+02:00
draft: false
categories: ["Python", "Data engineering"]
---

## 1. What is airflow and why would you use it?

* airflow manages dataflow as a graph (direct acyclic graph or *DAG*), which consists of separate tasks, and schedule them

> *Wait*, you may say, *I can do that with cron!*

Yes, you can, but with airflow:

* airflow implements a well-known `pipeline` design pattern with *DAG*s, which are data engineer's workhorses

* you can easily divide your app into smaller tasks and monitor their reliability and execution duration

* the performance is more transparent

* simple rerunning

* simple alerting with emails

* as the pipelines' definitions are kept in code, you can generate them, or even let the user do it

* you can (and should!) keep your pipelines' code in a git repository

* you keep the logs in one place (unless you have some central log repository in your company implemented using e.g. ELK)


## 2. Installation and startup

Installation in straightforward:

```bash
pip install apache-airflow
```

Then run airflow scheduler and webservice with:

```bash
airflow scheduler
airflow webserver
```
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin

## 3. Best practises

### Softlinks

You may feel tempted to create a git repository in your DAG folder, however this is not the best solution. It's much easier and more logical to keep your DAG file in a repo where your project lives and softlink it with

```bash
ln -s /path-to-your-project-repo/my_project_dag.py /home/me/airflow/dags/
```

### DAG names and DAG file names

* keep only one DAG in a file

* DAG should have the same name as the file it's in

* DAG's and file's name should begin with project's name

### Jinja templating

You can pass arguments to the command with jinja templating, instead of creating a command string by yourself. You can then keep all you parameters in a separate json file.

```{python, eval = FALSE}
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime


default_args = {
  'depends_on_past': False,
  'start_date': datetime(2018, 1, 15, 16, 45, 0),
  'email': ['test_mail@gamil.com'],
  'email_on_failure': False,
  'retries': 0
}

dag = DAG(
  'dag_name',
  schedule_interval='0 12 * * *',
  default_args=default_args,
  catchup=False
)

first_task = BashOperator(
  task_id='first_task',
  bash_command='echo {{ params.number }} {{ params.subparam.one }}',
  # you can keep params in a json file
  params={'number': '10', 'subparam' : {'one': '1'}},  
  dag=dag
)
```

## 3. Useful links

- [a good tutorial](http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow/)

- [yet another good tutorial](https://airflow.apache.org/tutorial.html)

- [deployig with airflow and containers](https://towardsdatascience.com/how-to-use-airflow-without-headaches-4e6e37e6c2bc)

Airflow's purpose is rather straightforward, so the best way to learn it is learning-by-doing.
