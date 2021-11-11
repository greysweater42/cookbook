---
title: "pillow"
date: 2021-11-10T16:04:17+01:00
draft: false
categories: ["Python", "Machine learning"]
---

## 1. What is `pillow` and why would you use it?

Pillow is a Python package for working with images. It is invaluable for working with image datasets, but personally I find its API rather hard to remember, so here I present a few of the most popular applications of pillow for machine learning.

A quick note on installation:

```{bash}
pip install pillow
```

## 2. Applications

### a) displaying an image in a VSCode interactive session 

>IMHO VSCode provides the best environment for data scientists at the moment.

This one is fairly straightforward:

```{python}
from PIL import Image
im = Image.open("image.jpg")
im
```

running simple `im` will display the image in the interactive session.
You can also run

```{python}
im.show()
```
which will open up a separate window with your image in it.

When you're done with looking at the image you should close it with:
```{python}
im.close()
```

which resembles working with files, but for them you usually use context managers (`with` statements). So to be always sure the image is closed you can run
```{python}
from IPython.display import display
from PIL import Image

with Image.open("image.jpg") as im:
    display(im)
```
but you have to specifically import `display` function, which will surely not be used in production.

### b) drawing on an image

There are various shapes you can draw on an image, but the most commonly used are lines, rectangles, polygons and text.

```{python}
from PIL import Image, ImageDraw, ImageColor
from IPython.display import display
with Image.open("image.jpg") as im:
    # check your image type (im.type) to see if it is RGB or black and white
    # if your image is in black and white, you will not be able to draw in color
    im = im.convert('RGB')  
    draw = ImageDraw.Draw(im)  # a layer used for drawing
    coords = (0, 0, 100, 100) # (x1, y1, x2, y2)

    fill = ImageColor.getrgb("yellow")  # you can also provide rgb as e.g. #add8e6
    draw.line(coords, fill=fill, width=10)  # draw a line
    draw.rectangle(coords, fill=fill)  # draw a rectangle
    draw.rectangle(coords, outline=fill, width=4)  # draw a border of rectangle

    # polygon
    poly_coords = [(0, 0), (100, 100), (100, 200), (0, 100)]
    draw.polygon(poly_coords, fill=fill)

    # polygon border - you have to repeat the first coord
    draw.line(poly_coords + [poly_coords[0]], fill="black", width=10)  # draw a line

    # semi-transparent rectangle / polygon
    im = im.convert('RGBA')  
    overlay = Image.new('RGBA', im.size)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(coords, fill=(0, 0, 0, 127))
    draw.polygon(poly_coords, fill=(200, 100, 0, 127))
    im = Image.alpha_composite(im, overlay)
    im = im.convert('RGB')  # re-conversion to RGB may be needed for saving as jpg

    # text
    draw = ImageDraw.Draw(im)  # resetting draw after semi-transparent rectangle
    draw.text((50, 50), "hello")
    display(im)

```

More about drawing you fill find in [pillow docs](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html).