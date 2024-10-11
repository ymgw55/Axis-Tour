FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
    
RUN apt-get update && apt-get install -y \
    git \
    curl \
    unzip \
    tar \
    build-essential \
    libsndfile1 \
    vim \
    libgl1-mesa-dev \
    libglib2.0-0 \
    wget \
    unzip

# Copy files from host to the image.
COPY requirements.txt /tmp/requirements.txt

# Install python package, remove copied file and cache.
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

# Create the user
ARG USERNAME
ARG USER_UID
ARG USER_GID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace