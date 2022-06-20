#!/bin/bash

rsync -rlptzv -e "ssh -p 1006" --progress --exclude=.git bracher@tunnel.bio.lmu.de:~/results/. ~/Documents/Studium/Master_NEURO/thesis/results
