---
title: "mlops"
date: 2021-02-17T16:56:08+01:00
draft: true
categories: ["scratchpad"]
---

> This is not a proper blog post yet, just my notes, hence it is in "scratchpad" category.

### 1. What and why

### 2. DVC

### 3. mlflow

> disclaimer: This is just my opinion. I believe that measuring performance in data science is extremely difficult. When we hire employees, we strive to quantify their skills as if it was possible to create an embedding of a person in a form of a vector or even a scalar. This is what the school does with students by giving them grades and calulating their average: then you can easily sort them by their... grades (not skills). I know that embeddings work well in several subjects, i.e. [natural language processing](https://greysweater42@github.io/nlp), but usually coming up with a proper measure/embedding is extremely difficult and data scientists tend to arbitrarily choose any measure (e.g. model accuracy) to optimize, without further reviewing whether it is sensible from the perspective of the company or not. It strucks me especially in the context of kaggle competitions, which many people believe that remain the best proxy/measure of the skill of a data scientist. Well, they don't. In fact, IMHO, people with very narrow specialization, like super-fancy cutting-edge state-of-the-art exotic deep naural networks trained on a huge cluster in a big data ecosystem ;), often apply this extremely complex methods into every problem they face, instead of taking a step back to get a broader perspective. Maybe even finding the solution of this problem, using a fancy algorithm with the best accuracy possible will not eventually give us what we wanted, e.g. attract new customers. Unfortunately, having no choice, employers oftem judge canidates by how "state-of-the-art" algorithms they've used recently, which in turn encourages data scientists to over-engineer their projects, which clearly exemplifies a conflict of interests between an employee and his/her employer. To summarize: IMHO, mlflow primarily helps in tuning the model, i.e. optimizing an abstract measure which may have absolutely no influence on your business. That is why, IMHO, you should really reconsider whether you need a specialized tool like mlflow instead of dvc, which does one or two general things, but does them extremely well. Don't over-enginner. Think wider.
> unless you want to use mlflow for kaggle.

