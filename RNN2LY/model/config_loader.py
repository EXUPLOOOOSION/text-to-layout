"""
loads the configuration for STRAN2LY model from a configuration json file
each checkpoint folder should have a configuration file
it does not load any weight.
"""
import json
from model.encoderRNN import PreEncoderRNN
from model.decoderRNN import DecoderRNN
from model.seq2seq import Seq2Seq
import sys
import traceback
import pickle
def generate_encoder(n, bidirectional=True, hidden_size=128, image_size=(256, 256)):
    """
    Function to generate the encoder
    """
    encoder = PreEncoderRNN(n, ninput=300, drop_prob=0.5, nhidden=hidden_size, nlayers=1, bidirectional=bidirectional)
    return encoder

def generate_decoder(vocab_size, is_training, use_attention=False, bidirectional=True, hidden_size=128, xy_distribution_size=16):
    """
    Function to generate the decoder
    """
    decoder = DecoderRNN(vocab_size, hidden_size, is_training, use_attention=use_attention, bidirectional=bidirectional, xy_distribution_size=xy_distribution_size)
    return decoder
def load_config(configuration_file_path):
    """
    Loads the necessary configurations to initialize a STRAN2LY model. returns the model and the configuration dict.
    Its initialized to eval mode. Training program should change it if needed.
    The configuration file should be PATH to a json file containing the following information:
    "freeze_encoder": Wether to freeze the encoder or not. true:not train it. false: train it Irrelevant in inference (evaluation)
    "pad_token": token to use as padding. 0 is appropiate for COCO
    "vocab_size": number of possible words in the vocabulary.
    "num_classes": number of possible output class tokens. Will be the dimension of class output for bboxes. num_classes+4 (<pad>, <sos>, <eos>, <unk>). 84 for COCO
    "hidden_size": size of feature embeddings outputted by encoder and used by decoder.
    "use_attention": whether the decoder should use attention.
    "xy_distribution_size": the xy coordinate probability grid will be of size (xy_distribution_size, xy_distribution_size)
    "bidirectional": whether the LSTM decoder should be bidirectional.
    "max_objs": maximum permitted objects in an scene. doesnt have <sos> and <eos> into account (will later be added by the program)
    "cap_lang_path": Only necessary when using generate(). A pickle containing a Lang object (from dataset.py) containing the vocab dictionary learned on training.
    """
    try:
        with open(configuration_file_path, 'r') as json_file:
            print("loading configuration from "+configuration_file_path)
            configuration = json.load(json_file)
    except FileNotFoundError:
        print("ERROR: Configuration file '"+str(configuration_file_path)+"' not found",file=sys.stderr)
        exit()
    except:
        print("Error while trying to read configuration file: ",file=sys.stderr)
        traceback.print_exc()
        exit()
    encoder, decoder = generate_encoder(configuration['vocab_size'], bidirectional = configuration['bidirectional'], hidden_size = configuration['hidden_size']), generate_decoder(configuration['num_classes'], False, xy_distribution_size=configuration['xy_distribution_size'], use_attention=configuration['use_attention'], hidden_size=configuration['hidden_size'],bidirectional=configuration['bidirectional'])
    # +2 objects because we need to include the <sos> and <eos>
    if 'cap_lang_path' in configuration.keys():
        try:
            with open(configuration['cap_lang_path'], 'rb') as f:
                cap_lang = pickle.load(f)
        except:
            print('file: '+configuration['cap_lang_path']+" non existant or without enough permission. Proceeding without cap_lang initialization on model")
            cap_lang = None
    else:
        cap_lang = None
    seq2seq = Seq2Seq(encoder, decoder, configuration['num_classes'],configuration['pad_token'], False, max_len=configuration['max_objs']+2, freeze_encoder=configuration['freeze_encoder'],cap_lang = cap_lang)
    return seq2seq, configuration
