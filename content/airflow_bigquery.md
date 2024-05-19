---
title: "airflow + BigQuery"
date: 2024-05-16T11:56:13+02:00
draft: false
categories: ["Data engineering"]
---


## Why would you use airflow with BigQuery?

Because they both work great for creating ELT processes, storing huge amounts of data, and providing a convenient interface to these data in SQL.

A couple of years ago I wrote a short blog post about [airflow 1]({{< ref "airflow" >}}). I also wrote about [BigQuery]({{< ref "terraform_bigquery" >}}) and how it works with `terraform`. You might want to have a look there as well.

## Prerequsites

Before you start working with airflow + GCP, you need to install and configure airflow (I recommend using virtualenv for that):

```bash
pip install apache-airflow
pip install 'apache-airflow[google]'

airflow inidb
export AIRFLOW_HOME=<wherever you want. config files and DAGS will be stored there>
```

After that you're ready to run airflow with:

```bash
airflow standalone
```

In this tutorial we use a local airflow setup. In production you would rather run airflow in docker containers on a remote server, on k8s cluster, or with Google Cloud Composer. From now on you can access airflow at http://127.0.0.1:8080.

Besides airflow, we need to set up GCP connection from our local computer:

```bash
gcloud auth application-default login
gcloud config set project bigquery-tutorial-xxxxxx
```

where `bigqeury-tutorial-xxxxxx` is GCP project's ID. 


## A brief example of how airflow and BigQuery work together

This is a very simple EL pipeline, which downloads some example data from the internet, uploads them to a GCP buckets and creates an external BigQuery table to enable users to access these data using SQL.

<img src="/airflow_bigquery.drawio.png" style="width: 100%;"/>

The following `cars.py` file should be stored in `$AIRFLOW_HOME/dags` folder.

```python
from datetime import datetime

from airflow import DAG
from airflow.operators import bash
from airflow.providers.google.cloud.operators import bigquery as bq, gcs
from airflow.providers.google.cloud.transfers import local_to_gcs


# this is a config external to airflow, so could be stored anywhere but here
params = {
    "bucket_name": "cars-xxxxxx",
    "dataset_name": "cars",
    "table_name": "cars",
    "tmp_file_path": "/tmp",
    "location": "EUROPE-CENTRAL2",
    "source_url": "https://raw.githubusercontent.com/abhionlyone/us-car-models-data/master/2013.csv",
}

# schema could be stored in a separate json file, e.g. in a bucket
cars_table_schema = [
    {"name": "year", "type": "integer"},
    {"name": "make", "type": "STRING"},
    {"name": "model", "type": "STRING"},
    {"name": "body_styles", "type": "STRING"},
]

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2018, 1, 15, 16, 45, 0),
    "email": ["test_mail@gmail.com"],
    "email_on_failure": False,
    "retries": 0,
}


dag = DAG("cars", default_args=default_args, catchup=False)

# the path should be stored somewhere else, probably as a parameter
download_file = bash.BashOperator(
    task_id="download_file",
    bash_command=f"wget {params['source_url']}",
    cwd=params["tmp_file_path"],
    dag=dag,
)

create_bucket = gcs.GCSCreateBucketOperator(
    task_id="create_bucket",
    bucket_name=params["bucket_name"],
    # in production we would rather use MULTI-REGION setting for reliability
    storage_class="REGIONAL",
    location=params["location"],
    dag=dag,
)

upload_file_to_bucket = local_to_gcs.LocalFilesystemToGCSOperator(
    task_id="upload_file_to_bucket",
    src=f"{params['tmp_file_path']}/2013.csv",
    dst="2013.csv",
    bucket=params["bucket_name"],
    dag=dag,
)

create_dataset = bq.BigQueryCreateEmptyDatasetOperator(
    task_id="create_dataset",
    dataset_id=params["dataset_name"],
    exists_ok=True,
    location=params["location"],
    dag=dag,
)

# file specific information could be stored somewhere else. Then we could reuse
# this DAG for downloading any data from the internet
create_table = bq.BigQueryCreateExternalTableOperator(
    task_id="create_external_table",
    destination_project_dataset_table=f"{params['dataset_name']}.{params['table_name']}",
    bucket=params["bucket_name"],
    source_objects=["*.csv"],
    schema_fields=cars_table_schema,
    source_format="CSV",
    skip_leading_rows=1,
    field_delimiter=",",
    quote_character='"',
    dag=dag,
)


(
    download_file
    >> create_bucket
    >> upload_file_to_bucket
    >> create_dataset
    >> create_table
)
```

## Remarks

- Airflow does not provide a convenient `destroy` functionality, which would allow us to easily get rid of all of the services that we provisioned.

- Airflow's BigQuery plugin does not have a simple "download data with a query functionality", which is a pity. But you can relatively easily create a custom one.

- In the architecture above we provision resources (bucket, bq dataset, bq table) and provide business logic (how we e.g. query and filter data) at the same time. You might want to have these two types of actions separately. This is why you might want to consider using `terraform` for creating infrastructure and DDL, and airflow for ELT only. There are also other tools available, like `dbt` or `DataForm`.

- Airflow has a declarative vibe, i.e. it appears you might define this DAG in a yaml file. There even is a [package which provides such a functionality](https://github.com/rambler-digital-solutions/airflow-declarative).

## Reading data from BigQuery

You can check if processing ended successfully in airflow GUI, but you might also want to query the data. There are many ways to do that, e.g. using GCP Console, GCP's CLI (`bq`) or from Python:

```Python
from google.cloud import bigquery

client = bigquery.Client()

# Perform a query.
QUERY = """
    SELECT * FROM `bigquery-tutorial-xxxxxx.cars.cars` 
    where make like 'Aston Martin';
"""
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish

df = rows.to_dataframe()
print(df)
```

*The example above was heavily insired by [BigQuery's documentation](https://cloud.google.com/bigquery/docs/samples/bigquery-query-results-dataframe).*

You can also use [pandas' special function for reading from BigQuery](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_gbq.html).


## Cleanup

After the pipeline ends successfully and you reviewed the results, you might want to delete the resources that the pipeline created in GCP:

```bash
gsutil -m rm -r gs://cars-xxxxxx
bq rm -f cars.cars
bq rm -f cars
```

## Resources

- [airflow's official guide to BigQUery operators](https://airflow.apache.org/docs/apache-airflow-providers-google/stable/operators/cloud/bigquery.html)

- [airflow's documentation](https://airflow.apache.org/docs/apache-airflow-providers-google/stable/_api/airflow/providers/google/cloud/operators/bigquery/index.html)

- [Data Pipelines with Apache Airflow](https://www.manning.com/books/data-pipelines-with-apache-airflow) - If I were to recommend only one resource to learn airflow, that would be this book.
