---
title: "shiny"
date: 2017-03-24T09:13:23+01:00
draft: false
categories: ["R"]
---

## 1. What is `shiny` and why would you use it?

* shiny is an R web framework designed for creating dynamic web applications without any knowledge of html, css and javascript, php etc. Pure R. 

Sounds like a dream?

* Advantages:

    * easy to learn the basics;

    * easy to set up.

* Disadvantages:

    * weak scalability;
    
    * poor performance;

    * in order to make the application work the way you want to, you have to involve javascript, html and css. This may be cumbersome, as the documentation is not very helpful;

    * shiny is a nieche framework, so it's community is little. Stackoverflow disappoints annoyingly often;

    * lack of a good book/tutorial. RStudio articles are not structured and I've spent hard time finding a passable learning path.

I've got mixed feelings about `shiny`. As a data scientist, maybe you should consider concentrating on fitting models and wrangling data instead of preparing shiny front-ends.

## 2. A "Hello World" example:

You can store your application in one file (e.g. "app.R"), like this:

```{r}
library(shiny)

inputBins <- 10

ui <- shinyUI(fluidPage(

  titlePanel("Hello Shiny!"),

  sidebarLayout(
    sidebarPanel(
      sliderInput("bins",
        "Choose your favourite number:",
        min = 1,
        max = 50,
        value = 30)
    ),

    mainPanel(
      plotOutput("distPlot")
    )
  )
))

server <- shinyServer(function(input, output) {

    r <- reactiveValues()

    observeEvent(input$bins, {
        r$bins <- input$bins
        output$distPlot <- renderPlot(f())
    })

    f <- function() {
        x <- faithful[, 2]
        inputBins <- r$bins
        bins <- seq(min(x), max(x), length.out = inputBins  + 1)
        hist(x, breaks = bins, col = 'darkgray', border = 'white')
    }

})

shinyApp(ui, server)
```

and run it with

```{r}
Rscript app.R
```

or divide it into two separate files:

ui.R
```{r}
library(shiny)

ui <- shinyUI(fluidPage(

  titlePanel("Hello Shiny!"),

  sidebarLayout(
    sidebarPanel(
      sliderInput("bins",
        "Choose your favourite number:",
        min = 1,
        max = 50,
        value = 30)
    ),

    mainPanel(
      plotOutput("distPlot")
    )
  )
))
```

server.R
```{r}
library(shiny)

inputBins <- 10

server <- shinyServer(function(input, output) {

    r <- reactiveValues()

    observeEvent(input$bins, {
        r$bins <- input$bins
        output$distPlot <- renderPlot(f())
    })

    f <- function() {
        x <- faithful[, 2]
        inputBins <- r$bins
        bins <- seq(min(x), max(x), length.out = inputBins  + 1)
        hist(x, breaks = bins, col = 'darkgray', border = 'white')
    }

})
```

and run it with
```{r}
Rscript -e "shiny::runApp()"
```

or from within the R console

```{r}
shiny::runApp()
```

## 3. Useful links

* [a one-file app](https://shiny.rstudio.com/gallery/single-file-shiny-app.html)

* [a two-files app](https://shiny.rstudio.com/articles/basics.html)

* [various shiny widgets](https://shiny.rstudio.com/gallery/widget-gallery.html)

* [shinytest](https://stackoverflow.com/questions/36338779/unit-testing-shiny-apps)
