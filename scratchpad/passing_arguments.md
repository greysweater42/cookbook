from
https://stackoverflow.com/questions/1009860/how-to-read-process-command-line-arguments

https://docs.python.org/2/library/argparse.html

https://mkaz.blog/code/python-argparse-cookbook/
```
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="write report to FILE", metavar="FILE")
parser.add_argument("-q", "--quiet",
                    action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")

args = parser.parse_args()
```