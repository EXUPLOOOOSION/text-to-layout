"""
loads the configuration for STRAN2LY model from a configuration json file
each checkpoint folder should have a configuration file
it does not load any weight.
"""
import json
from model.encoderBERT import EncoderBERT
from model.decoderTRAN import DecoderTRAN
from model.seq2seq import Seq2Seq
import sys
import traceback

def generate_encoder():
    """
    Function to generate the encoder
    """
    encoder = EncoderBERT()
    return encoder

def generate_decoder(num_classes, input_size, nhead, num_decoder_layers, dim_FFN, class_hidden_layers,xy_hidden_layers,wh_hidden_layers, xy_distribution_size=16, temperature=0.4):
    """
    Function to generate the decoder
    """
    decoder = DecoderTRAN(num_classes, input_size, nhead, num_decoder_layers, dim_FFN, class_hidden_layers,xy_hidden_layers,wh_hidden_layers, xy_distribution_size=xy_distribution_size, temperature=temperature)
    return decoder


def load_config(configuration_file_path):
    """
    Loads the necessary configurations to initialize a STRAN2LY model. returns the model and the configuration dict.
    Its initialized to eval mode. Training program should change it if needed.
    The configuration file should be PATH to a json file containing the following information:
    "num_classes": number of possible output class tokens. Will be the dimension of class output for bboxes. num_classes+4 (<pad>, <sos>, <eos>, <unk>). 84 for COCO
    "hidden_size": size of feature embeddings outputted by encoder and used by decoder. depends on the underlying sentence encoder. 768 for BERT
    "nhead": number of heads the decoder-transformer will use.
    "num_decoder_layers": number of transformer layers the decoder will use.
    "xy_distribution_size": the xy coordinate probability grid will be of size (xy_distribution_size, xy_distribution_size)
    "temperature": temperature to use in tempearture softmax in xy coordinate sampling.
    "freeze_encoder": Wether to freeze the encoder or not. true:not train it. false: train it Irrelevant in inference (evaluation)
    "max_objs": maximum permitted objects in an scene. doesnt have <sos> and <eos> into account (will later be added by the program)
    "dim_FFN": the dimension of the feedforward network model inside each Decoder Layer.
    "class_hidden_layers": List. Number of neurons in the hidden layers of the FFN that has to get the class probabilities of each bbox from the embeddings outputted by the transformer decoder.
                            The total network will have [hidden_size, [class_hidden_layers], num_classes] shape
    "xy_hidden_layers": List. Number of neurons in the hidden layers of the FFN that has to get the xy distribution probabilities of each bbox from the embeddings outputted by the transformer decoder.
                            The total network will have [hidden_size, [xy_hidden_layers], xy_distribution_size**2] shape
    "wh_hidden_layers": List. Number of neurons in the hidden layers of the FFN that has to get the width and height of each bbox from the embeddings outputted by the transformer decoder.
                            The total network will have [hidden_size, [wh_hidden_layers], 2] shape
    """
    try:
        with open(configuration_file_path, 'r') as json_file:
            print("loading configuration from "+configuration_file_path)
            configuration = json.load(json_file)
    except FileNotFoundError:
        print("ERROR: Configuration file '"+str(configuration_file_path)+"' not found",file=sys.stderr)
        exit()
    except:
        print("Error while trying to read configuration file" + configuration_file_path +": ",file=sys.stderr)
        traceback.print_exc()
        exit()
    encoder, decoder = generate_encoder(), generate_decoder(configuration['num_classes'], configuration['hidden_size'], configuration['nhead'], configuration['num_decoder_layers'],configuration['dim_FFN'], configuration['class_hidden_layers'],configuration['xy_hidden_layers'],configuration['wh_hidden_layers'], xy_distribution_size=configuration['xy_distribution_size'], temperature=configuration['temperature'])
    # +2 objects because we need to include the <sos> and <eos>
    seq2seq = Seq2Seq(encoder, decoder, configuration['num_classes'], False, max_len=configuration['max_objs']+2, freeze_encoder=configuration['freeze_encoder'])
    return seq2seq, configuration