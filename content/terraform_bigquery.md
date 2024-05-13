---
title: "terraform + bigquery"
date: 2024-05-13T07:50:45+02:00
draft: false
categories: ["DevOps", "Data engineering"]
---


## Why would you use `terraform` in data engineering?

`terraform` is a standard in industry for setting up cloud infrastructure. You might wonder, just as I do, whether you should store your DDLs in `terraform`: it certainly is possible, but there are some drawbacks, e.g. how to handle database schema migrations? And how does it cooperate with other tools, like [Apache Airflow](({{< ref "airflow" >}}))? Surely there are interesing alternatives, like [dbt](https://www.getdbt.com/) or [dataform](https://cloud.google.com/dataform?hl=pl).

In this article I will not judge whether `terraform` is a good choice for your particular use case. But it could be.

## Prerequisites

#### GCP project

A "project" is GCP's abstraction to group resources and it has one billing account (one debit card attached). In GCP you always work inside of a project, so we need to create one. It is very simple: go to the GCP Console, type "projects" into the search bar, choose "Create a project", provide its name and click "Create". 

For this tutorial I created a project named `bigquery-tutorial`.

*Theoretically we could create project from terraform, but you usually set up payment methods manually, because you do not change them ever after programatically. It's not developer's job.*

#### gcloud

GCP provides a nice CLI for working with its services: `gcloud`, and terraform uses it to provision infrastructure. In order to install it, follow [these](https://cloud.google.com/sdk/docs/install) instructions. 

After the installation you need to log in to your Google account with `gcloud auth login`. You might also need to run `gcloud auth application-default login`, and `gcloud auth application-default set-quota-project bigquery-tutorial`.


## terraform config

#### setup

*For primer on `terraform`, please refer to [this article]({{< ref "terraform_ec2_vpc#terraform-primer" >}})*

```terraform
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

locals {
  region = "EUROPE-CENTRAL2"
}

provider "google" {
  project = "bigquery-tutorial-<id>"
  region  = local.region
}
```

For this tutorial I chose Europe-central2 region (Warsaw, Poland), because this is where I live. After running `terraform init` terraform installs plugins that it needs to work with GCP.

#### bucket

Creating a bucket is very simple:

```terraform
resource "google_storage_bucket" "bucket" {
  name          = "some-bucket"
  location      = local.region  # one-region bucket
  force_destroy = true

  public_access_prevention = "enforced"
}
```

You might notice some interesting config options here:

- `location` - buckets on GCP are usually replicated across multiple regions for additional reliability and efficiency. In this case, since it's an example, we use a single region
- `force_destroy` - if bucket is not empty, `terraform` will recreate/destroy it anyway. This is not a default behavior.
- `public_access_prevention` - normally you don't want the bucket to be accessible from the internet, unless you host a static webpage. Public access prevention can be set either to `inherited`, whre it inherits this setting from organization's policy, or `forced`, where it overrides organization's policy and prevents the access.

For further options you might want to refer to the [documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket), among which are:

- multi-region (us / eu / asia) / dual-region / region
- storage class (standard, nearline, coldline, archive)
- access control - uniform / fine-grained
- data protection (soft delete, object versioning, retention)
- encryption (google managed, CMEK)

## BigQuery

`terraform` allows you to create BigQuery datasets and tables:

```terraform
resource "google_bigquery_dataset" "default" {
  dataset_id = "cars_per_year"
  location = local.region  # one-region dataset
}

resource "google_bigquery_table" "default" {
  dataset_id = google_bigquery_dataset.default.dataset_id
  table_id   = "cars"
  schema = file("./cars_schema.json")
  # we want to be able to recreate this table anytime we modify its config
  deletion_protection = false

  external_data_configuration {
    autodetect    = false
    source_format = "CSV"

    csv_options {
        quote = "\""  
        skip_leading_rows = 1  # skips header
    }

    source_uris = [
      "${google_storage_bucket.bucket.url}/*.csv",
    ]
  }

  depends_on = [
    google_bigquery_dataset.default,
  ]
}
```

where we read tables schema from a `cars_schema.json` file:

```json
[
  {
    "name": "year",
    "type": "integer"
  },
  {
    "name": "make",
    "type": "STRING"
  },
  {
    "name": "model",
    "type": "STRING"
  },
  {
    "name": "body_styles",
    "type": "STRING"
  }
]
```


## Why `terraform` might be not the best choice for BigQuery?

- terraform creates buckets, datasets, and tables without any regard to the data. It is a little bit too easy to delete data
- you might want to keep your DDLs next to data processing pipelines, which enables you to use CTAS (`create table ... as ...`) syntax. Besides, whenever you modify a pipeline, you might modify a table and it is easier to do that with one tool (e.g. `dbt`)
- you might want to create buckets and datasets with `terraform`, but tables with some other tool
