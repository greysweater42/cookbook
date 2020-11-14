---
title: "xgboost"
date: 2020-11-14T13:34:53+01:00
draft: false
categories: ["Machine learning"]
---

## 1. What is XGBoost and why would you use it?

- xgboost is one of the most popular supervised learning algorithms

- it's popularity results from a few factors:

    - it's easy to use

    - is available in R and Python, with sklearn api

    - usually performs very well even on default hyperparameters

    - is resilient to missing values and ourliers

    - provides excellect results

    - is very fast, comparing to other methods with similar prediction capabilities

    - works well for both regression and classification problems

## 2. Why an article about xgboost?

Despite it's broad popularity, few people (due to my personal experience) really understand how xgboost works. This may result from:

- objective difficulty of the subject

- xgboost requires quite a broad knowdledge of machine learning

- it is a relatively new algorithm, so data scientists with 5+ years of experience did not have a chance to learn in at school/university

- tutorials available on the internet are:

    - too obscure, e.g. Friedman's legendary paper presenting gradient boosting is, imho, extremely hard to grasp

    - too general, i.e. they present only the "intuition behind" and how to use xgboost library in Python, without even mentioning any mathematical concepts underneath

    - just a copy-and-paste from Friedman's paper withour any explanation... wchich reminds me of submitting an assignment by copy-pasting some equations from Wikipedia (this is just my personal opinion. There is nothing wrong with not understanding machine learning, but we don't have to pretend we do if we don't.)

## 3. How did I approach understanding xgboost?

First of all, I tried to read Friedman's paper on Gradient Boosting and I completely failed. Then I started searching for tutorials on towardsdatascience and medium, but they were far from being in-depth, as I can already use xgboost Python api and have a basic intuition of how xgboost works.

Many papers and articles later I finally managed to get a better understanding of xgboost, mainly thanks to these resources:

- [explained.ai and it's tutorial on gradient boosting](https://explained.ai/gradient-boosting/index.html) - a perfect place to start. Gradient boosting and tree boosting are basic concepts used in xgboost, so deep understanding of them is crucial.

- [legendary Friedman's article on gradient boosting](https://projecteuclid.org/euclid.aos/1013203451) - a super difficult paper, but explained.ai clarifies many concepts in it

- [explained.ai regularization](https://explained.ai/regularization/index.html) - as xgboost uses regularization to avoid overfitting, you should have a good understanding of this method

- random forests - any tutorial should do, this is a rather straight-forward algorithm

- [xgboost paper](https://arxiv.org/abs/1603.02754) - having read all the previous materials, now you are ready to deep dive into this paper. 

Good luck! :)
