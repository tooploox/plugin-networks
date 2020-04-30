#!/bin/bash

WORKSPACE_DIR="workspace/sun397"


fileId=1S_7eulwWjdzHYbyjezLMwZSjH8EbjF9O
fileName=plugin_networks_workspace.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

tar xvf ${fileName}
rm ${fileName}

if [[ ! -f SUN397.tar.gz ]]; then
    wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
fi

if [[ ! -d ${WORKSPACE_DIR}/images ]]; then
    tar xvf SUN397.tar.gz
    mv SUN397 ${WORKSPACE_DIR}/images
fi

