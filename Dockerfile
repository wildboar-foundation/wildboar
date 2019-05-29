FROM quay.io/pypa/manylinux2010_x86_64
MAINTAINER Isak Samsten
VOLUME ["/io"]
COPY build-wheel.sh /build-wheel.sh
ENTRYPOINT ["/bin/bash", "/build-wheel.sh"]


