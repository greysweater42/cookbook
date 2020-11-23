# basic programs

sudo apt-get update

sudo apt-get install vim -y
sudo apt-get install tmux -y
sudo apt-get install curl -y
sudo apt-get install apt-transport-https -y
sudo apt-get install git -y

# echo export LC_CTYPE=en_US.UTF-8 >> .bashrc
# echo export LC_ALL=en_US.UTF-8 >> .bashrc
# source .bashrc

echo "imap jk <Esc>" > .vimrc

# java
sudo apt update
sudo apt install openjdk-8-jdk openjdk-8-jre -y


# sudo apt-get update
# sudo apt-get -y install software-properties-common

# echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/java-8-debian.list
# echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/java-8-debian.list

# echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections
# echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EEA14886
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886
# sudo apt-get update
# sudo apt-get -y install oracle-java8-installer

# elasticsearch
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-key update  # fixes weird error with 'authenication'
sudo apt-get update && sudo apt-get install -y elasticsearch 
# 
# # if run in VM, before running elasticsearch, check if vm has more than 4GB of RAM (manually in virtualbox)
# 
sudo update-rc.d elasticsearch defaults 95 10
sudo -i service elasticsearch start 

# verify if elasticsearch is working (it may take a few seconds for elasticsearch to start)
# curl http://127.0.0.1:9200

# in order to use elasticsearch via vm, configure:
echo "network.bind_host: 0" | sudo tee -a /etc/elasticsearch/elasticsearch.yml
echo "network.host: 0.0.0.0" | sudo tee -a /etc/elasticsearch/elasticsearch.yml
echo "http.port: 8881" | sudo tee -a /etc/elasticsearch/elasticsearch.yml

echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install kibana -y

echo "server.host: 0.0.0.0" | sudo tee -a /etc/kibana/kibana.yml
echo "server.port: 8882" | sudo tee -a /etc/kibana/kibana.yml

sudo update-rc.d kibana defaults 95 10
sudo -i service kibana start 

# http://192.168.33.10:9200
# http://192.168.33.10:5601

# logstash
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install logstash -y

sudo service logstash start

# unbind C-b
# set-option -g prefix C-a
# bind-key C-a send-prefix
# 
# ## https://gist.github.com/spicycode/1229612
# bind-key v split-window -h -c "#{pane_current_path}"
# bind-key s split-window -v -c "#{pane_current_path}"
# 
# bind-key J resize-pane -D 5
# bind-key K resize-pane -U 5
# bind-key H resize-pane -L 5
# bind-key L resize-pane -R 5
# 
# bind-key M-j resize-pane -D
# bind-key M-k resize-pane -U
# bind-key M-h resize-pane -L
# bind-key M-l resize-pane -R
# 
# # Shift arrow to switch windows
# bind -n S-Left  previous-window
# bind -n S-Right next-window
# 
# # No delay for escape key press
# set -sg escape-time 0

# https://logz.io/blog/securing-elk-nginx/
sudo apt-get install nginx -y
sudo apt-get install apache2-utils -y

sudo htpasswd -c /etc/nginx/htpasswd.users kibanauser

JaszczompLeciGurom
JaszczompLeciGurom
