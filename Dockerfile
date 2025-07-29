# syntax=docker/dockerfile:1
# [Dockerfile reference guide](https://docs.docker.com/go/dockerfile-reference/)

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# # Create a non-privileged user that the app will run under.
# # See https://docs.docker.com/go/dockerfile-user-best-practices/
# # --> This results in permssion denied problems in matplotlib and 
# #     bellatrex, as they try to write temp files to /nonexistent and 
# #     /usr/local/lib/... respectively
# ARG UID=10001
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --shell "/sbin/nologin" \
#     --no-create-home \
#     --uid "${UID}" \
#     appuser

# # Switch to the non-privileged user to run the application.
# USER appuser

# Copy the source code into the container.
COPY . .

# Set the DEPLOYED environment variable
ENV DEPLOYED=True

# Expose the port that the application listens on.
EXPOSE 8050

# Run the application.
CMD python src/app.py
