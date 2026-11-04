# ROS 2 Humble, MAVROS and ArduCopter SITL. ArduCopter is pinned to the version
# the suite was validated on, because parameter names change between releases.
FROM ros:humble-ros-base

ARG COPTER_VERSION=4.7.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-mavros ros-humble-mavros-extras ros-humble-message-filters \
      python3-pip curl ca-certificates libatomic1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sfL https://raw.githubusercontent.com/mavlink/mavros/ros2/mavros/scripts/install_geographiclib_datasets.sh -o /tmp/geo.sh \
    && bash /tmp/geo.sh

RUN pip3 install --no-cache-dir "numpy<2" onnx onnxruntime

RUN mkdir -p /opt/sitl \
    && curl -sfL "https://firmware.ardupilot.org/Copter/stable-${COPTER_VERSION}/SITL_x86_64_linux_gnu/arducopter" -o /opt/sitl/arducopter \
    && curl -sfL "https://raw.githubusercontent.com/ArduPilot/ardupilot/Copter-${COPTER_VERSION}/Tools/autotest/default_params/copter.parm" -o /opt/sitl/copter.parm \
    && chmod +x /opt/sitl/arducopter

WORKDIR /work
