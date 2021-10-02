---
title: "reading mate"
date: 2021-09-27T00:32:01+02:00
draft: false
categories: ["Projects"]
---


## 1. Is it OK to judge people by the books they read?

Well, not really. Personally, from time to time I read complete crap purposedly, out of curiosity and to form my own opinion about it (I didn't like 50 Shades of Gray as much as I had expected) and sometimes I read books which may seem completely contradictory to my views/beliefs, because I try to be open and give them a chance to convince me (which sometimes happens, but e.g. Osho's meditation techinques were a little too extreme for me).

But still, anytime I visit someone and I see a plentiful bookshelf, I can't help myself but analyse it thouroughly and make a judgment on my host. Sorry for being superficial.

Who knows, maybe this person would be my reading mate?

## 2. Is there any way to automate finding new reading mates?

Wow, this is going to be a true [programmer move](https://img.devrant.com/devrant/rant/r_2516404_bkZxN.jpg).

Theoretically, I could take a picture of my friend's bookshelf with my smartphone, and the app that I wrote would:

* transform an image into a list of books

* compare this list to a list of books that I read and tell how many of these book we both read

* or tell me something more about his/her personality, if there is any research on connection between personality and books of your choice.

In this particular blog post I will stick to the first dot only, because smartphone app development is not really my cup of tea, and the last dot is super difficult and... as I wrote at the beginning, I wouldn't really believe it. 

## 3. Reading text from an image

is a fairly popular machine learning problem, known as OCR, or Optical Character Recognition. Let's move on to the specifics:

* This time I want to get my hands dirty with [keras](https://greysweater42.github.io/keras), because by now I have never used it in practice.

* There are a few OCR problems that are solved using keras from which I can learn, like [captcha reading](https://keras.io/examples/vision/captcha_ocr/) which is an excellent example when machine learning engineers make new jobs in programming

* and a few keras-based python packages supporting OCR, including [keras-ocr](https://github.com/faustomorales/keras-ocr).

## 4. Solution

Task turned out to be quite simple, so I didn't have tweak, not even know keras at all. Here's ahwt I had to do:
- read an image from the internet

<div style="overflow: hidden;padding: 10px;">
    <img src="/reading_mate/raw_image.jpg" style="width: 100%;"/>
</div>

- use `keras_ocr` to find letters/words in this imamge
- segregate these words, so they made book titles

Here comes the code:

```{python}
import matplotlib.pyplot as plt
import keras_ocr
import numpy as np


def read_image():
    url = "https://pbs.twimg.com/media/E7zIo5RXMAQ9kEV.jpg"
    image = keras_ocr.tools.read(url)  # 1
    im = np.flip(image.transpose((1, 0, 2)), 0)  # rotate image counterclockwise
    return im


def read_titles(pred):
    books = []
    coords_max = 0
    for text in pred:
        letters, coords = text
        if min(coords[:, 1]) > coords_max:
            if books:
                books[-1] = [x for _, x in sorted(zip(order, books[-1]))]
            order = []
            books.append([])
        books[-1].append(letters)
        coords_max = max(coords[:, 1])
        order.append(min(coords[:, 0]))

    titles = [" ".join(b).upper() for b in books]
    return [t for t in titles if len(t) > 10]


def plot_predictions(im, pred):
    _, axs = plt.subplots(nrows=1, figsize=(20, 20))
    return keras_ocr.tools.drawAnnotations(image=im, predictions=pred, ax=axs)


def main():
    im = read_image()
    pipeline = keras_ocr.pipeline.Pipeline()  # 2
    pred = pipeline.recognize([im])[0]  # 3
    plot_predictions(im, pred)
    print(read_titles(pred))  # 4

main()
```

The code is, IMHO, surpisingly short if we considered how difficult the task actually is. Most of the work is done by `keras_ocr` package, but a few parts may seem a little confusing:
- #1 I used `keras_ocr.tools.read(url)`, a function provided by `keras_ocr`, to read image from the internet
- #2 predictions are made using a Pipeline, which I defined in this place
- #3 this is how the pipeline is executed to make a prediction
- #4 `read_titles` is a simple function, which groups words by books, creating titles and authors

## 5. Results

<div style="overflow: hidden;padding: 10px;">
    <img src="/reading_mate/output.png" style="width: 100%;"/>
</div>

```
['SPUFFORD FRANCIS LIGHI PERPETUAL', 
'S  GREAT CIRCLE', 
'SUNJEEV SAHOTA CHINA ROOM', 
'BEWILDERMENT RICHARD POWERS', 
'THE JORTUNE MEN MOHAMED NADIFA', 
'NO ONE IS TALKING ABOUT THIS PATRICIA LOCKWOOD', 
'MARY LAWSON A TOWN CALLED SOLACE', 
'AN ISLAND KAREN JENNINGS', 
'KAZUO ISHIGURO KLARA AND THE SUN', 
'TIR SWEETNESS O WATER NATHA HARRIS', 
'THE PROMISE DAMON GALGUT', 
'RACHELCUSK SECOND PLACE', 
'ARUDPRAGASAM ANUK NORTH PASSAGE A']
```

`keras_ocr` clearly can read some text, but not perfectly:
- it has problems with reading rotated text, that is why before the prediction I rotated the picture by 90 degrees counterclockwise
- sometimes it simply misreads the text, which is fine: no model is perfect. Some mistakes though are quite ambiguous, take the fifth book from the top: *The Fortune Man*. The "F" could be easliy misread by a human as well, while others result rather from the font used: *RACHELCUSK* (third from bottom) looks like one word, doesn't it? To sum up: the algorithm ususally works just fine in *normal* cases and makes mistakes in those cases, where human might also have problems.

Another issue is that this algorithm cannot tell the difference between the author and the title, which is obvious; I sometimes can't tell it myself either.

## 6. Summary

It was quite fun to use OCR, but it seems that reading is a little more that just recognizing letters, e.g. you have to know:
- the direction of the text (quite obvious for human)
- little bit about the language, e.g. know, that *Rachel* is a name
- the language you recognize. In this particular case we can tell the author from the title, but what if the books were not in English, but in some exotic language (like Polish? ;)) and we know there is know word like *juture*, but there is something like *future*, and *J* looks a little like *F*

All of these problems can be addressed programatically, i.e. solved, but it would require much more sophisticated software.