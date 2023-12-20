

This script was made to classify dialects, comparing the effectiveness between the angles of the vowel triangle
In 'information.py' is the location of the:

+ ===> directories needed: tmit_path, data_path, output_path
+ ===> name of the model/attempt
+ ===> list of information:
    - phones, vowels, 'all_info'
+ ===> to reduce phones/group phones: group_phones, reduce_groups
+ ===> multiple dictionaries that change the following:
    - MFCC: n_mels, hoplength, framelength \n
    - Design: delete_dialects, delete_gender, 'selection of phones in input and output', 
    - input classifiers: select frames, delta, double delta 
    - network: classes to classify (dialects, gender, phones), layers, batch size, epoch 

############################################################################### \n
paths: 
- tmit_path(string): "home/user/Documents/timit"
- data_path(string): "home/user/Documents/research/data"
- output_path(string): "home/user/research/output"

############################################################################### \n
* ===> tmit_path
- timit files: this assumes that in this directory is a TEST and a TRAIN directory, in which both PHN and WAV files are located (unseperated)
so, the location of timit_dir/TEST should have DR1FAKS0SA1.PHN and DR1FAKS0SA1.wav


* ===> data_path
+ is the directory where the dataframes to csv are saved. (This could do with much more optimization, currently,
every dataframe is saved in its whole.)
MFCC = mfcc variables, normal is 13_160_400 see below
    -   In the main file: (MFCC_TRAIN.TXT; MFCC_TEST.TXT)
    -   In a directory MFCC: middles_index_TEST & middles_index_TRAIN; for every key in dictionary 'selected' (see below): test & train.txt
    -   In a directory MFCC/model_N: possibly a sel_TEST/TRAIn.txt, information.txt which has all parameters, output.txt which has classification report and
    for every key a comparison matrix as csv.

* ===> output_path
- is the directory where system links for every model is made.
Here: a model means a set of parameters, the selection of phones can be in the same model.

###############################################################################
information: 
- phones(list) = a list with all the phones
            -  vowels(list) = a list with all vowels in timit
             - all_info(dict) = a dictionary with groupings of vowels as keys (such as 'vowels', 'all_u') and phones as values
             - group_data(dict) = a dictionary with all phones as keys, and the grouping it should go to as value ('ax-h': 'a')
             - reduce_groups(boolean) = determines if the above dictionary is applied
###############################################################################

* ===> all_info: this script was made to compare different group of phones (such as 'all_a sounds', 'all i sounds') and the
effectiveness of this on dialect recognition. This is a dictionary that groups these sounds, but it also allows for
making different models later on (see INPUT DESIGN, selection, below) This is useful because grouping data does not allow
for one value being in multiple groups (for example, 'IH' is in 'just_i', 'all_i' and 'vowels')


* ===> group_data is a dictionary that groups phones (some phones are not interesting, for phone recognition, some are
grouped. This is a way you can do that. Set 'reduce_groups = True')


The models can be varied as followed / the different features:
###############################################################################
+ > MFCC:
        -  n_mels (int) = number of bands of the mfcc
        -  hoplength(int) = hoplength in frames, in timit, this is 1/16 ms, so hoplength of 160 = 10 ms
        - framelength(int) = framelength in frames
###############################################################################
In mfcc_to_df.py is the script to create a dataframe with meta information and mfccs from the
wav files in timit. The width depends on n_mels.
It is structured as follows:
  -   ### dataframe with each row:
   -  ### column 0 = dialect
   -  ### column 1 = gender
   - ### column 2 = speaker ID
   - ### column 3 = sentence type
   - ### column 4 = sentence number
   - ### column 6 = phone
   - ### column 7 = frame in wav file
   - ### column 8: = mfcc, width depending on n_mels

All of these variables are directly inserted into the librosa.mfcc.

###############################################################################
+ - > input classifiers:
                      delta(bool) = add delta
                      double_delta(bool) = add double_delta
                      select_frames(None, bool, int, list) = selects middle frame(s)
###############################################################################
In input_classifiers.py, there are functions to transform the data:
if we take frame x, then delta adds columns with the mfcc data from (x+1)-(x-1), call this d
double_delta adds columns with (d+1)-(d-1) or ((x+2)-x)-(x-(x-2))
select_frames will select the middle frames of a segment of a phone. You can give a list, in which the first int will be
the number of frames before the middle frame, and the second int will be the number of frames after the middle frame.
So, if the data is [1,2,3,4,5],
- [0,0] will return 3 ([1,2,3,4] will return 2);
- [1,1] will return [2,3,4];
- [1,2] will return [2,3,4,5]
- n will return the same as [n,n]
- true will return [1,1]


###############################################################################
* -> input design:
                delete_dialects(list) = which dialects to not count
                delete_gender(string or none) = which genders to not count
                selection(dic) = which phones to count

###############################################################################

delete_dialects will remove all dialects from that list from the data before running the network. This is in integers:
dialect 1 in timit is also dialect 1 here. (So dialect 0 does not exist)
delete_gender will remove the selected gender from the data before running the network. This is in string. Can be
'F', 'M' or None. if delete_gender = ['F'] , the network will only run on males.
selection is a dictionary, which has a name (for example 'all_vowels') and a list of phones (see information.py).
It will determine how many models to make. Each entry in the dictionary will create its own models and will create
its own output con. matrix.

###############################################################################
* -> network design:
              -  type(str or list) = network type
               - epoch(int) = number of epochs
              -    batch_size(int) = number of batch sizes
              -  network_classes(int) = what are the classes the network goes to? Standard = 0
###############################################################################

*All network is done in networking.py
there are 2 types of network: standard, which can be muted, and cnn, which cannot be muted. Cnn is currently wide ->
1 mfcc of data turns into
type(list) expects a list of the following type: [['relu','relu'][32,16]] so: a list of names of activation layers,
and a list of sizes that these activation layers should be. currently, the last layer is always the size of the output
classification, and a SOFTMAX.
- Network_classes: see above for what column corresponds to what. dialect = 0, gender = 1, phone = 5


OUTPUT:

Not yet implemented in Main.
Checks all folders in output path, makes all confusion matrices from the data in that folder
To_excel makes an excel file with all the output. Each model has its own sheet. 
