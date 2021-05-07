#!/bin/bash
export PATH="$HOME/anaconda3/bin:$PATH"
source activate neoism
cd $HOME/neoism
python3 lstm_generate_brainfuck.py -I 192.168.1.110 -sP 7770 -S softmax --arduino-serial-number 7573530323135161F1D2 data/neoism.txt neoism_l2_n64,1024_d0.1,0.2_em5_b512/lstm-weights--layers2-nhu64,1024-e054.model.hdf5 > /home/tats/neoism/neoism.log 2> /home/tats/neoism/neoism_err.log
