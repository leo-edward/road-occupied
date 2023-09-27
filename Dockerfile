FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

USER root
# Change debian source to China mirror
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY sources.list /etc/apt/sources.list
# Set the time zone to China
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' >/etc/timezone

# apt updata and install dependencies
RUN apt update \
&& apt install -y freetds-dev \
&& apt install -y unixodbc-dev \
&& apt install -y libgl1-mesa-glx \
&& apt install -y openssh-server \
&& apt install -y openssh-client \
&& apt install -y vim \
&& apt install -y libgl1-mesa-glx \
&& apt install -y libglib2.0-0 \
&& apt install -y zip

# mkdir /yolov7-client
RUN mkdir /yolov7-client
# copy yolov7-client to /yolov7-client
COPY . /yolov7-client
# set workdir to /yolov7-client
WORKDIR /yolov7-client
# Change the pipy source to China mirror
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# install python dependencies
RUN pip install -r requirements.txt
# run python main.py
# CMD ["python3", "main.py"]
# run with gunicon
CMD ["gunicorn", "-c", "gunicorn.conf.py", "manage:app"]
