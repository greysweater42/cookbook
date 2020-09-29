FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository -y "ppa:marutter/rrutter3.5"
RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install r-base=3.6.3-1bionic -y

RUN Rscript -e "sessionInfo()"

RUN apt-get install libssl-dev -y
RUN apt-get install libcurl4-openssl-dev -y
RUN apt-get install libxml2-dev -y
RUN apt-get install liblapack-dev -y
RUN apt-get install gfortran -y
RUN apt-get install pandoc=1.19.2.4~dfsg-1build4 -y
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip

RUN Rscript -e 'install.packages("remotes")'
COPY dependencies /dependencies
WORKDIR /dependencies
RUN Rscript installer.R

RUN Rscript -e 'devtools::install_github("tomis9/decisionTree")'
RUN Rscript -e 'devtools::install_github("vqv/ggbiplot")'
RUN Rscript -e "blogdown::install_hugo(version = '0.52')"

RUN pip3 install -r requirements.python.txt

RUN mkdir -p /cookbook
WORKDIR /cookbook

CMD /bin/bash /cookbook/deploy.sh prod build
