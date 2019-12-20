# Main routine for HMM training/decoding
# T. Hueber - CNRS/GIPSA-lab - 2019
# MASTER MOSIG/MSIAM - FDPM 
########################################
from __future__ import print_function
import numpy as np
import scipy
import scipy.io as sio
import os
from os import listdir, mkdir, system
from os.path import join, isdir, basename, splitext
import pdb # debugger

# CONFIG
#########
htk_bin_dir = "htk/bin/linux/";

train_scp = "data/train.scp";
test_scp = "data/test.scp";
corpus_mlf = "data/Thomas_lemonde_1_150_aligned.mlf";

phonelist = "data/phonelist";
hmm_proto = "models/proto";

# training options
nIte = 10; 

# test options
dict_filename = "lm/dict_phones";
grammar = "lm/grammar_phones";
wip = -20;
test_mlf = "data/Thomas_lemonde.mlf";

# Step 
step_train = 1;
step_hmm_gmm = 0;
step_test = 0;

if step_train:
    # TRAINING  
    ##########
    
    # Computing variance floors
    print("Computing variance of all features (variance flooring)\n");
    if isdir("models/hmm_0")==False:
        mkdir("models/hmm_0"); 

    system(htk_bin_dir + "/HCompV -T 2 -f 0.01 -m -S " + train_scp + " -M models/hmm_0 -o models/average.mmf " + hmm_proto);
    
    # Generating hmm template
    system("head -n 1 " + hmm_proto + " > models/hmm_0/init.mmf");
    system("cat models/hmm_0/vFloors >> models/hmm_0/init.mmf");

    # HMM parameters estimation using Viterbi followed by Baum-welch (EM) alg.


    all_phones = [line.rstrip() for line in open(phonelist)]

    if isdir("models/hinit")==False:
        mkdir("models/hinit/");

    if isdir("models/hrest")==False:
        mkdir("models/hrest/");

    for p in range(np.shape(all_phones)[0]):

        print("===============" + all_phones[p] + "================\n");
        system(htk_bin_dir +"/HInit -T 000001 -A -H models/hmm_0/init.mmf -M models/hinit/ -I " + corpus_mlf + " -S " + train_scp + " -l "+ all_phones[p] + " -o " + all_phones[p] + " " + hmm_proto)

        system(htk_bin_dir + "/HRest -A -T 000001 -H models/hmm_0/init.mmf -M models/hrest/ -I " + corpus_mlf + " -S " + train_scp + " -l " + all_phones[p]+ " models/hinit/" + all_phones[p])


    # Making monophone mmf
    # load variance floor macro
    f=open("htk/scripts/lvf.hed","w")
    f.write("FV \"models/hmm_0/vFloors\"\n");
    f.close();

    if isdir("models/herest_0")==False:
        mkdir("models/herest_0")

    system(htk_bin_dir + "/HHEd -d models/hrest/ -w models/herest_0/monophone.mmf htk/scripts/lvf.hed " + phonelist);

    # HMM parameter generation using embedded version of Baum-welch algorithm
    for i in range(nIte):
        if isdir("models/herest_" + str(i+1))==False:
            mkdir("models/herest_" + str(i+1))

	# embedded reestimation
        system(htk_bin_dir + "/HERest -A -I " + corpus_mlf + " -S " + train_scp + " -H models/herest_" + str(i) + "/monophone.mmf -M models/herest_" + str(i+1) + " " + phonelist);



    # Make a copy of last model parameters
    system("cp models/herest_" + str(i+1) + "/monophone.mmf models/herest_" + str(i+1) + "/monophone_gmm.mmf")

    if step_hmm_gmm:
        # Increase number of gaussians per state up to 2
        f = open("./hhed.cnf","w");
        f.write("MU %i {*.state[2-4].mix}\n" % 2);
        f.close()

        system(htk_bin_dir + "/HHEd -A -H models/herest_" + str(i+1) + "/monophone_gmm.mmf ./hhed.cnf " + phonelist);

        # Re-estimate model parameters (let's do 5 iterations of EM)
        for r in range (5):
            system(htk_bin_dir + "/HERest -T 0 -S " + train_scp + " -H models/herest_" + str(i+1) + "/monophone_gmm.mmf -I " + corpus_mlf + " " + phonelist)

            # ... up to 4
            f = open("./hhed.cnf","w");
            f.write("MU %i {*.state[2-4].mix}\n" % 4);
            f.close()

            system(htk_bin_dir + "/HHEd -A -H models/herest_" + str(i+1) + "/monophone_gmm.mmf ./hhed.cnf " + phonelist);

            # Again, we re-estimate model parameters (let's do 5 iterations of EM)
            for r in range (5):
                system(htk_bin_dir + "/HERest -T 0 -S " + train_scp + " -H models/herest_" + str(i+1) + "/monophone_gmm.mmf -I " + corpus_mlf + " " + phonelist)
# TEST
if step_test:
    # convert grammar rules to decoding network
    system(htk_bin_dir + "/HParse " + grammar + " lm/wnet")

    # phonetic decoding using Viterbi algorithm (Token Passing)
    system(htk_bin_dir + "/HVite -y lab -p " + str(wip) + " -m -T 1 -S " + test_scp + " -H models/herest_" + str(nIte) + "/monophone_gmm.mmf -i data/rec.mlf -w lm/wnet " + dict_filename + " " + phonelist)

    # Calculate WER 
    system(htk_bin_dir + "/HResults -A -X rec -s -t -f -I " + corpus_mlf + " " + phonelist + " data/rec.mlf")



    #################
