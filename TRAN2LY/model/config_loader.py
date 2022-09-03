"""
loads the configuration for STRAN2LY model from a configuration json file
each checkpoint folder should have a configuration file
it does not load any weight.
"""
import json
from model.encoderBERT import EncoderBERT
from model.decoderRNN import DecoderRNN
from model.seq2seq import Seq2Seq
import sys
import traceback

def generate_encoder(freeze_encoder=False, pooling_type='max'):
    """
    Function to generate the encoder
    """
    encoder = EncoderBERT(None, freeze_encoder, pooling_type = pooling_type)
    return encoder

def generate_decoder(vocab_size, is_training, use_attention=False, bidirectional=True, hidden_size=128, xy_distribution_size=16, temperature = 0.4):
    """
    Function to generate the decoder
    """
    decoder = DecoderRNN(vocab_size, hidden_size, is_training, use_attention=use_attention, bidirectional=bidirectional, xy_distribution_size=xy_distribution_size, temperature = temperature)
    return decoder


def load_config(configuration_file_path):
    """
    Loads the necessary configurations to initialize a STRAN2LY model. returns the model and the configuration dict.
    Its initialized to eval mode. Training program should change it if needed.
    The configuration file should be PATH to a json file containing the following information:
    "freeze_encoder": Wether to freeze the encoder or not. true:not train it. false: train it Irrelevant in inference (evaluation)
    "pad_token": token to use as padding. 0 is appropiate for COCO
    "num_classes": number of possible output class tokens. Will be the dimension of class output for bboxes. num_classes+4 (<pad>, <sos>, <eos>, <unk>). 84 for COCO
    "hidden_size": size of feature embeddings outputted by encoder and used by decoder. depends on the underlying sentence encoder. 768 for the default 'sentence-transformers/all-mpnet-base-v2'
    "use_attention": whether the decoder should use attention.
    "xy_distribution_size": the xy coordinate probability grid will be of size (xy_distribution_size, xy_distribution_size)
    "bidirectional": whether the LSTM decoder should be bidirectional.
    "temperature": temperature to use in tempearture softmax in xy coordinate sampling.
    "max_objs": maximum permitted objects in an scene. doesnt have <sos> and <eos> into account (will later be added by the program)
    "pooling_type": how the encoder pools each word's embedding into a single sentence embedding. 'max', 'avg','cls'
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
    encoder, decoder = generate_encoder(configuration['freeze_encoder'],pooling_type=configuration['pooling_type']), generate_decoder(configuration['num_classes'], False, xy_distribution_size=configuration['xy_distribution_size'], use_attention=configuration['use_attention'], hidden_size=configuration['hidden_size'],temperature=configuration['temperature'])
    # +2 objects because we need to include the <sos> and <eos>
    seq2seq = Seq2Seq(encoder, decoder, configuration['num_classes'],configuration['pad_token'], False, max_len=configuration['max_objs']+2, freeze_encoder=configuration['freeze_encoder'])
    return seq2seq, configuration