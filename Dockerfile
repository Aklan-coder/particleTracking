# .oni extraction environment — works natively on Apple Silicon (arm64)
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenni2-dev python3 python3-pip python3-numpy python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install openni --break-system-packages

WORKDIR /work
COPY extract_oni.py /work/extract_oni.py

# Usage: docker run --rm -v $(pwd):/data oni-extract /data/recording.oni /data/out
ENTRYPOINT ["python3", "/work/extract_oni.py"]
