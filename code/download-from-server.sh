#!/bin/bash
#hallo
rsync -rlptzv -e "ssh -p 1024" --progress --delete --exclude=.git bracher@tunnel.bio.lmu.de:~/code/. ~/Documents/Studium/Master/thesis/code/
