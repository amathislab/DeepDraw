#!/bin/bash

# Enters into docker container and then starts running do_data run for DeepDraw models
# Kai Sandbrink, 2020.11.09

# idea: https://stackoverflow.com/questions/35547966/how-to-write-a-bash-script-which-automate-entering-docker-container-and-doing

docker start kai_dlcdocker_data
docker exec --user kai -i kai_dlcdocker_data bash <<'EOF'
cd /media/data/DeepDraw/DeepDraw/single_cell
python3 controls_main.py --S True --data True
EOF
