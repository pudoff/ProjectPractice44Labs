# ODA File Converter (free "guest" download from opendesign.com) headless in Xvfb.
# Build: docker build -t oda-convert docker/oda-convert
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xvfb xauth \
      libglib2.0-0t64 libgl1 libegl1 libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
      libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
      libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxcb-xfixes0 libxrender1 libxi6 libsm6 libice6 \
      libopengl0 libxcb-glx0 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/oda.deb "https://www.opendesign.com/guestfiles/get?filename=ODAFileConverter_QT6_lnxX64_11dll.deb" \
    && (apt-get update && apt-get install -y --no-install-recommends /tmp/oda.deb || (dpkg -i /tmp/oda.deb; apt-get -f install -y)) \
    && rm -rf /var/lib/apt/lists/* /tmp/oda.deb
COPY convert.sh /usr/local/bin/convert.sh
RUN chmod +x /usr/local/bin/convert.sh
WORKDIR /work
