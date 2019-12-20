# Extract MFCC using HTK HCopy tools
# T. Hueber - CNRS/GIPSA-lab 
# ENSIMAG - Lecture on speech technology
########################################
#!/usr/bin/perl
# 
use File::Basename;

# CONFIG
#########
$audio_dir = "data/wav16";
$mfcc_dir = "data/mfcc/";
$hcopy_bin = "htk/bin/macosx/HCopy";
$hcopy_config_file = "htk/config/mfcc.cnf";

# RUN 
#####
opendir(AUDIODIR, $audio_dir);
my @all_audio_filenames = readdir(AUDIODIR);
closedir(AUDIODIR);

foreach my $current_audio_filename (@all_audio_filenames)
{
    # skip . and ..
    next if($current_audio_filename =~ /^\.$/);
    next if($current_audio_filename =~ /^\.\.$/);

    # $file is the file used on this iteration of the loop
    
    print "Encoding $current_audio_filename \n";

# get basename
    $basename = basename($current_audio_filename,".wav");
    $target_feature_filename = "$mfcc_dir/$basename.mfcc";
    system("$hcopy_bin -C $hcopy_config_file $audio_dir/$current_audio_filename $target_feature_filename");

}

## END of extract_mfcc.pl
#########################
