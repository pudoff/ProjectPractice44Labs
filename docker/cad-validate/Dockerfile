# Headless CAD validation: LibreCAD dxf2pdf -> PNG preview (+ optional LibreDWG if built in).
# Build:  docker build -t cad-validate docker/cad-validate
# Run:    docker run --rm -v "<abs dir with dxf>:/work" cad-validate validate.sh /work/file.dxf
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    QT_QPA_PLATFORM=offscreen \
    LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends librecad fonts-dejavu-core poppler-utils ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY validate.sh /usr/local/bin/validate.sh
RUN sed -i 's/\r$//' /usr/local/bin/validate.sh && chmod +x /usr/local/bin/validate.sh
WORKDIR /work
