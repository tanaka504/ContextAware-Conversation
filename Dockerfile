FROM ubuntu:16.04
MAINTAINER tanaka504 <nishikigi.nlp@gmail.com>

# apt-get
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install git vim curl locales mecab libmecab-dev mecab-ipadic-utf8 make xz-utils file sudo bzip2 wget python3-pip swig
RUN apt-get -y install libssl-dev libbz2-dev libreadline-dev libsqlite3-dev 
# install pyenv
ENV HOME /root
RUN git clone https://github.com/yyuu/pyenv.git $HOME/.pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    eval "$(pyenv init -)"

RUN pyenv install 3.6.0
RUN pyenv global 3.6.0

# install python3 packages
RUN pip3 install --upgrade pip
RUN pip3 install mecab-python3

# character encoding
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# install CRF++
WORKDIR /usr/src/
RUN wget 'https://docs.google.com/uc?export=download&id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ' -O CRF++-0.58.tar.gz
RUN tar xf CRF++-0.58.tar.gz
WORKDIR CRF++-0.58/
RUN ./configure --prefix=$HOME/usr
RUN make && make install


# install CaboCha
RUN curl -sc /tmp/gcokie "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU" > /dev/null
RUN getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
RUN curl -Lb /tmp/gcokie "https://drive.google.com/uc?export=download&confirm=${getcode}&id=0B4y35FiV1wh7SDd1Q1dUQkZQaUU" -o cabocha-0.69.tar.bz2
RUN tar xvf cabocha-0.69.tar.bz2
RUN cd cabocha-0.69
RUN export LDFLAGS="-L$HOME/usr/lib"
RUN export CPPFLAGS="-I$HOME/usr/include"
RUN ./configure --with-mecab-config=$HOME/usr/bin/mecab-config --with-charset=UTF8 --prefix=$HOME/usr
RUN make && make install

# python setup.py build
# python setup.py --user install

# add MeCab Neologd
WORKDIR /usr/src/
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git /usr/src/mecab-ipadic-neologd && \
/usr/src/mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y
RUN mecab -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/


WORKDIR /home/
CMD ["/bin/bash"]
