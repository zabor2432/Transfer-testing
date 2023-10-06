FROM python:3.7

ARG OPENMPI_VERSION="4.1.4"
ARG OPENMPI_MAJOR_VERSION="v4.1"
ARG OPENMPI_CONFIGURE_OPTIONS
ARG OPENMPI_MAKE_OPTIONS="-j4"


# Download, build, and install OPENMPI
RUN mkdir /tmp/openmpi-src
WORKDIR /tmp/openmpi-src
RUN wget https://download.open-mpi.org/release/open-mpi/${OPENMPI_MAJOR_VERSION}/openmpi-${OPENMPI_VERSION}.tar.gz \
      && tar xfz openmpi-${OPENMPI_VERSION}.tar.gz
RUN cd openmpi-${OPENMPI_VERSION} && ./configure ${OPENMPI_CONFIGURE_OPTIONS}
RUN cd openmpi-${OPENMPI_VERSION} && make all ${OPENMPI_MAKE_OPTIONS}
RUN cd openmpi-${OPENMPI_VERSION} && make install
RUN rm -rf /tmp/openmpi-src


#### ADD DEFAULT USER ####

ARG USER=mpi
ENV USER ${USER}

RUN useradd -rm -d /home/${USER} -s /bin/bash -g root -G sudo ${USER}
ENV USER_HOME /home/${USER}

#### CREATE WORKING DIRECTORY FOR USER ####

ARG WORKDIR=/project
ENV WORKDIR ${WORKDIR}
RUN mkdir ${WORKDIR}
RUN chown -R ${USER}:root ${WORKDIR}

WORKDIR ${WORKDIR}
USER ${USER}

COPY dnn-topology .
RUN pip install -r requirements.txt

#### BUILD CMAKE ####

USER root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2.tar.gz
RUN tar -zxvf cmake-3.24.2.tar.gz
WORKDIR ${WORKDIR}/cmake-3.24.2
RUN ls
RUN ./bootstrap
RUN make
RUN make install
#RUN apt install cmake
RUN ldconfig
USER ${USER}
WORKDIR ${WORKDIR}

#### CLONE AND BUILD DIPHA ####

RUN git clone https://github.com/DIPHA/dipha.git

WORKDIR ${WORKDIR}/dipha
RUN mkdir build
WORKDIR ${WORKDIR}/dipha/build
RUN cmake -H. ..
RUN make 

WORKDIR ${WORKDIR}
#COPY dnn-topology . 

ENTRYPOINT ["python","./main.py"]
