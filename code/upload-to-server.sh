#!/bin/bash

rsync -rlptzv -e "ssh -p 1006" --progress --delete --exclude=.git ~/Documents/Studium/Master_NEURO/thesis/code/. bracher@tunnel.bio.lmu.de:~/code/
