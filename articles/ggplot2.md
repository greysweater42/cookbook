---
title: "ggplot2"
date: 2017-03-24T09:03:49+01:00
draft: false
categories: ["R"]
tags: ["R", "ggplot2", "tidyverse"]
---






## 1. What is ggplot2 and why would you use it?

* ggplot2 is an R package which makes creating nice-looking plots easy;

* the plots you create are highly customisable;

Once you learn ggplot2, you will not make any production plots using basic R. However, due to it's verbosity, for simple exploratory analysis I still use basic functions: plot, lines, hist and boxplot.

## 2. A few "Hello World" examples

### Basic plots

Let's define some sample data that we will work on:

```r
sample_data <- data.frame(
  a = letters[1:10], 
  b = sample(x = 1:10, size = 10),
  color = sample(x = c("red", "green", "blue"), size = 10, replace = TRUE)
)

print(sample_data)
```

```
##    a  b color
## 1  a  7  blue
## 2  b  2  blue
## 3  c  1  blue
## 4  d  6 green
## 5  e  5 green
## 6  f  3  blue
## 7  g  4  blue
## 8  h  8   red
## 9  i  9   red
## 10 j 10 green
```

The most basic plot:

```r
library(ggplot2)
ggplot(data = sample_data, mapping = aes(x = a, y = b)) + 
    geom_point()
```

![plot of chunk unnamed-chunk-2](./media/ggplot2/unnamed-chunk-2-1.png)

A little bit less basic plot, as points can be categorised by their colors:

```r
ggplot(data = sample_data, mapping = aes(x = a, y = b, color = color)) +
    geom_point()
```

![plot of chunk unnamed-chunk-3](./media/ggplot2/unnamed-chunk-3-1.png)

As you can see, colors do not match their descriptions, but you can customise it.

Here's another way of separating categories:

```r
ggplot(data = sample_data, mapping = aes(x = a, y = b)) + 
    geom_point() +
    facet_wrap(~ color, nrow=1)
```

![plot of chunk unnamed-chunk-4](./media/ggplot2/unnamed-chunk-4-1.png)

### Combining multiple types of plots

We'll use a dataset `mpg` which is available in ggplot2 package.

```r
ggplot(data = mpg) + 
    geom_smooth(mapping = aes(x = displ, y = hwy, color = drv, linetype=drv)) +
    geom_point(mapping = aes(x = displ, y = hwy, color = drv))
```

```
## `geom_smooth()` using method = 'loess' and formula 'y ~ x'
```

![plot of chunk unnamed-chunk-5](./media/ggplot2/unnamed-chunk-5-1.png)

Smoothing may be useful if you want to show trend.

You can have different mapping for every plot:

```r
ggplot(data = mpg, mapping = aes(x = displ, y = hwy)) +
    geom_point(mapping = aes(color = class)) +
    geom_smooth()
```

```
## `geom_smooth()` using method = 'loess' and formula 'y ~ x'
```

![plot of chunk unnamed-chunk-6](./media/ggplot2/unnamed-chunk-6-1.png)

Different datasets are also possible, but rather unusual.

### Bar plots

The simplest bar plot:

```r
ggplot(data = sample_data) +
    geom_bar(mapping = aes(x = a, y = b), stat = "identity")
```

![plot of chunk unnamed-chunk-7](./media/ggplot2/unnamed-chunk-7-1.png)

We had po provide the argument `stat = "identity"`, becasue the default behaviour is to plot the size/count of every category (x).

Stacked bar plot:

```r
ggplot(data = sample_data, mapping = aes(x = color, y = b, color = a)) +
    geom_bar(stat = "identity", fill = NA)
```

![plot of chunk unnamed-chunk-8](./media/ggplot2/unnamed-chunk-8-1.png)

### Boxplot

Let's use mpg data again:

```r
ggplot(data = mpg, mapping = aes(x = class, y = hwy)) +
    geom_boxplot() +
    coord_flip()
```

![plot of chunk unnamed-chunk-9](./media/ggplot2/unnamed-chunk-9-1.png)

We also used `coord_flip()`, which rotates the plot by 90 degrees, or, another words, flips the coordinates.


### Maps

Let's draw quickly a map of the USA:

```r
usa <- map_data("usa")
ggplot(usa, aes(long, lat, group = group)) +
    geom_polygon(fill = "white", color = "black") +
    coord_quickmap()
```

![plot of chunk unnamed-chunk-10](./media/ggplot2/unnamed-chunk-10-1.png)

We used two interesting functions:

* `map_data()` - a ggplot2's function, which provides spatial data for a few countries in the world;

* `coord_quickmap()` - adjusts the size of a plot to the size of map. Default settings cause the opposite.


### Other functions example

Here's a weird plot, which aims at presenting various customisation examples:

```r
ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) +
    geom_rect(mapping=aes(xmin=15, xmax=20, ymin=0, ymax=max(hwy)), 
              fill='blue', alpha=0.1) +
    geom_point() + 
    labs(title = "Some plot",
         subtitle = "subtitle to chart",
         caption = "and caption: made by me",
         x = "city miles per gallon", 
         y = "highway miles per gallon") +
    geom_abline(color ="red") +  
    theme_bw() 
```

![plot of chunk unnamed-chunk-11](./media/ggplot2/unnamed-chunk-11-1.png)

```r
    theme(plot.title = element_text(hjust = 0.5, size=12),
          axis.title = element_text(size=12))
```

```
## List of 2
##  $ axis.title:List of 11
##   ..$ family       : NULL
##   ..$ face         : NULL
##   ..$ colour       : NULL
##   ..$ size         : num 12
##   ..$ hjust        : NULL
##   ..$ vjust        : NULL
##   ..$ angle        : NULL
##   ..$ lineheight   : NULL
##   ..$ margin       : NULL
##   ..$ debug        : NULL
##   ..$ inherit.blank: logi FALSE
##   ..- attr(*, "class")= chr [1:2] "element_text" "element"
##  $ plot.title:List of 11
##   ..$ family       : NULL
##   ..$ face         : NULL
##   ..$ colour       : NULL
##   ..$ size         : num 12
##   ..$ hjust        : num 0.5
##   ..$ vjust        : NULL
##   ..$ angle        : NULL
##   ..$ lineheight   : NULL
##   ..$ margin       : NULL
##   ..$ debug        : NULL
##   ..$ inherit.blank: logi FALSE
##   ..- attr(*, "class")= chr [1:2] "element_text" "element"
##  - attr(*, "class")= chr [1:2] "theme" "gg"
##  - attr(*, "complete")= logi FALSE
##  - attr(*, "validate")= logi TRUE
```

## 3. Curiosities

### plotly

If you want to publish a plot on your website, consider using `plotly`:


```r
p <- ggplot(sample_data, aes(x=a, y=b)) +
    geom_point()

plotly::ggplotly(p)
```

![plot of chunk unnamed-chunk-12](./media/ggplot2/unnamed-chunk-12-1.png)

as it will give your plot interesting interactive features. Shiny users will appreciate them.

### ssh

When you work on a remote machine and connect to it via ssh, the plots you create will not appear in pop-up windows by default. In order to do this, add the `-X` flag when connecting to server:


```bash
ssh -X user@login
```

## 4. Useful links

* [R for Data Science](https://r4ds.had.co.nz/data-visualisation.html) - a very good book for plotting in ggplot2. A few examples in this tutorial were inspirded by it.

* [R Graphics Cookbook](http://www.cookbook-r.com/Graphs/) - looking for a quick answer? This is the right place for you.