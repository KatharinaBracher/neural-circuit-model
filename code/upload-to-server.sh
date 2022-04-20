#!/bin/bash

rsync -rlptzv -e "ssh -p 1024" --progress --delete --exclude=.git ~/Documents/Studium/Master/thesis/code/. bracher@tunnel.bio.lmu.de:~/code/
