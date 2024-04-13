---
title: "Fourier transform"
date: 2021-09-16T15:44:46+02:00
draft: false
categories: ["Machine learning"]
---

## 1. What is Fourier transform and when can it be useful?

Fourier transform is a method for representing a signal, time series or any other data than can be ordered (not necessarily chronologically, but this is usually the case) in a way where we can clearly see frequencies/seasonality.

Fourier discovered (not sure if mathematicians actually *invent* or *discover* stuff) that any signal can be represented as a sum of different sinuses and cosinuses. The differences between them are their frequencies.

The most common way of plotting a signal is with time on x axis and values/magnitude/measurements on y axis. It is useful for several purposes, e.g. we can see the trend and changes in variance over time. But signals are sometimes additive functions of many signals 

$$ y_i = \sum_{k=1}^{K} x_{ki} $$

some of which may be seasonal. Fourier transform can extract the seasonality from the data, which afterwards can be treated as the interesting part of the data or the noise, e.g. in financial data we may want to get rid of the seasonality, while in brainwaves the "waves" are what interests us the most.

## 2. Understanding the concept

Fourier transform is quite an advanced piece of mathematics. I recommend starting with the following subbjects:

[Lockdown math with 3Blue1Brown](https://www.youtube.com/watch?v=ppWPuXsnf1Q&list=PLZHQObOWTQDP5CVelJJ1bNDouqrAhVPev)

- imaginary numbers. Even if you are familiar with them, this will be an excellent introduction to Euler's formula.

[Imaginary numbers and how to see them](https://www.youtube.com/watch?v=T647CGsuOVU&list=PLiaHhY2iBX9g6KIvZ_703G3KJXapKkNaF&index=1)

- Euler's formula. If you have ever studied mathematics at the university, you probably have already heard about it. In this case you would have to understand it a little bit better.

- Fourier transform https://www.youtube.com/watch?v=1JnayXHhjlg


## 3. How I understand the concept

Being an econometrics graduate (which stands for statistics in economics) I find this flow of reasoning the most appealing:

* Fourier transform resembles linear regression models, where the exogenic variables are all possible sinus functions:

$$ y_t = \beta_1 \sin(t \cdot 2\pi) + \beta_2 \sin(t \cdot 2\pi / 2) + \beta_3 \sin(t \cdot 2\pi / 3) + ...$$

* or in short

$$ y_t = \sum_{k=1}^{K} \beta_k \sin(t \cdot 2\pi / k) $$

* linear regression for comparison:

$$ y_t = \sum_{k=1}^{K} \beta_k x_{kt} + \epsilon_t $$

so linear regression also has as error term. Fourier transform, in this case: Discrete Fourier Transform, as we are dealing with a sum of sinusoids, not an integral over all posiible sinusoids, can match perfectly to the data, without an error. In other words, you can represent any time series of length *t* with a sum of *t* sinusoids of various frequencies.
