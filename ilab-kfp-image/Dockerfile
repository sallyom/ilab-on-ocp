FROM registry.redhat.io/rhel9/buildah
ADD containerfile-ilab-kfp /tmp/input/Dockerfile
ADD build.sh /usr/bin
RUN chmod a+x /usr/bin/build.sh
ENTRYPOINT ["/usr/bin/build.sh"]
