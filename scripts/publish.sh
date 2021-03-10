#! /bin/bash
cd ..
export DOCKER_IMAGE="quay.io/pypa/manylinux1_x86_64"
export PLAT="manylinux1_x86_64"
sudo docker pull $DOCKER_IMAGE
sudo docker run --rm -e PLAT=$PLAT -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/scripts/build_wheels.sh
