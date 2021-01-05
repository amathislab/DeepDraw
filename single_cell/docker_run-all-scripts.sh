#!/bin/bash

# Enters into docker container and then starts running do_data run for DeepDraw models
# Kai Sandbrink, 2020.11.09

# idea: https://stackoverflow.com/questions/35547966/how-to-write-a-bash-script-which-automate-entering-docker-container-and-doing

docker start kai_dlcdocker_data_tf15
docker exec --user kai -i kai_dlcdocker_data_tf15 bash <<'EOF'
cd /media/data/DeepDraw/DeepDraw/single_cell
python3 controls_main.py --ST True --data True
exit
EOF
docker stop kai_dlcdocker_data_tf15
python3 controls_main.py --ST True --results True
python3 controls_main.py --ST True --analysis True
