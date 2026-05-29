1. git clone ... this_repo
2. cd this_repo
3. git submodule init
4. git submodule update
5. mamba env create -n env mamba_env.yml
6. conda activate env
7. cd cratertools ; pip install .  ; cd ..
6. copy the gebco file into the source_data/gebco directory
7. python lowres/scripts/global.py
8. run monitor.bash to process the files as they are generated, probably in a loop (for i in `seq ...`) or a cron job
9. 