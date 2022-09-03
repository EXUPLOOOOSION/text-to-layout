import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class DecoderTRAN(nn.Module):
    
    def __init__(self, num_classes, input_size, nhead, num_decoder_layers, dim_FFN, class_hidden_layers,xy_hidden_layers,wh_hidden_layers,dropout_p=0.1, xy_distribution_size=16, temperature=0.4, verbose=True):
        """
        Args:
            num_classes: int. number of possible output class tokens having sos and eos into account. Will be the dimension of class output for bboxes.
            input_size: int. size of vectors to be imputted as memory (context). In the case of using a BERT encoder to get the context vectors: 768
            nhead: int. number of attention heads each layer of the Decoder Transformer will have
            num_decoder_layers: int. number of attention layers the Decoder Transformer will have
            dim_FFN: int. dimension of Decoder Transformer feed forward network.
            class_hidden_layers: python list. number of neurons of each FF layer for the induction of the class from the output embedding vector of each bbox
            xy_hidden_layers: python list. number of neurons of each FF layer for the induction of the x y coordinates from the output embedding vector of each bbox
            wh_hidden_layers: python list. number of neurons of each FF layer for the induction of the width and height from the output embedding vector of each bbox
            dropout_p: float. dropout probability for the entire model (both transformer and FF networks)
            xy_distribution_size: int. size of the grid in which the objects are placed in. The grid will have a size of xy_distribution_size*xy_distribution_size
            temperature: temperature to use in tempearture softmax in xy coordinate sampling.
            verbose: whether to print information about the model when initializing it.
        """

        #Other possible parameters: transformer's feedforward dim dim_feedforward, transformer's activation function
        super(DecoderTRAN, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout_p
        self.xy_distribution_size = xy_distribution_size
        self.temperature = temperature
        # input and output size of the transformer model. Has to be dividable by 3. Should be the same as the encoder-transformer output_size
        self.input_size = input_size
        
        #remember to use dropout in these.
        #input_size = 768. Which is dividable by 3 to 256. These will later be concatenated to make a single input_size vector
        self.class_emb = nn.Embedding(self.num_classes, self.input_size//2,padding_idx=0) 
        self.class_dropout = nn.Dropout(p=self.dropout)
        self.xy_emb = nn.Linear(2,self.input_size//4)
        self.xy_dropout = nn.Dropout(p=self.dropout)
        self.wh_emb = nn.Linear(2,self.input_size//4)
        self.wh_dropout = nn.Dropout(p=self.dropout)
        #model to get embeddings for each object in the image
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_size, nhead=nhead, dropout=self.dropout, dim_feedforward=dim_FFN#, batch_first = True
            ), num_layers=num_decoder_layers
        )

        #model to get class probability distribution from object embedding + previous class embedding
        layers_class_out = [self.input_size+self.input_size//2]+class_hidden_layers+[self.num_classes]
        self.class_out = build_mlp(layers_class_out)
        
        #model to get xy probability distribution from object embedding + current class one-hot encoding
        layers_xy_out = [self.input_size+self.num_classes]+xy_hidden_layers+[self.xy_distribution_size**2]
        self.xy_out = build_mlp(layers_xy_out)
        
        #model to get wh from object embedding + current class one-hot encoding + previous wh embedding
        layers_wh_out = [self.input_size + self.num_classes + self.input_size//4]+wh_hidden_layers+[2]
        self.wh_out = build_mlp(layers_wh_out, final_nonlinearity=False)
        self.wh_sigmoid = nn.Sigmoid()
        if verbose:
            #print information about the initialization
            print()
            print("Transformer Decoder Initialized")
            
            print("\tnhead: ", nhead)
            print("\tnum_decoder_layers: ", num_decoder_layers)
            print("\thidden_size: ", input_size)
            print("\tvocab_size: ", num_classes)
            print("\ttemperature: ",self.temperature)
            print("\tDecoder Layer FFN Dimension: ", dim_FFN)
            print("\tlayers_class_out: ",layers_class_out)
            print("\tlayers_xy_out: ",layers_xy_out)
            print("\tlayers_wh_out: ",layers_wh_out)
    def train(self):
        self.is_training = True
    def eval(self):
        self.is_training = False
        
    def forward(self, target_l, target_x, target_y, target_w, target_h, target_key_mask, encoder_output, encoder_output_key_mask):
        target_x = target_x.unsqueeze(2)
        target_y = target_y.unsqueeze(2)
        target_w = target_w.unsqueeze(2)
        target_h = target_h.unsqueeze(2)
        # target_l = [batch_size, seq]
        # target_x, y, w, h = [batch_size, seq, 1]

        
        # Concatenate the [x, y, w, h] coordinates
        target_xy = torch.cat((target_x, target_y), dim=2)
        target_wh = torch.cat((target_w, target_h), dim=2)

        embedded_l = self.class_dropout(self.class_emb(target_l))
        embedded_xy = self.xy_dropout(self.xy_emb(target_xy))
        embedded_wh = self.wh_dropout(self.wh_emb(target_wh))
        
        tgt = torch.cat((embedded_l, embedded_xy, embedded_wh), dim=2)
        
        #fuck old nn.TransformerDecoderLayer. All my homies hate old nn.TransformerDecoderLayer without batch_first = True
        #en versiones mas nuevas de pytorch, inicializa nn.TransformerDecoderLayer con batch_first = True y los permutes de abajo no hacen falta. el bool tampoco
        
        tgt = tgt.permute(1,0,2)
        encoder_output = encoder_output.permute(1,0,2)
        encoder_output_key_mask = encoder_output_key_mask.bool() == False
        
        #print(target_key_mask)
        decoder_emb = self.decoder(tgt,encoder_output, memory_key_padding_mask = encoder_output_key_mask, tgt_key_padding_mask = target_key_mask)
        #still hate it
        decoder_emb = decoder_emb.permute(1,0,2)

        class_out_input = torch.cat((decoder_emb, embedded_l), dim=2)
        class_prob = self.class_out(class_out_input)
        class_prob = F.softmax(class_prob, dim=2).clamp(1e-5, 1)
        predicted_class = class_prob.argmax(2)
        class_one_hot = F.one_hot(predicted_class, num_classes = self.num_classes)

        xy_out_input = torch.cat((decoder_emb, class_one_hot), dim=2)
        xy_prob = self.xy_out(xy_out_input)
        #xy_distance = F.log_softmax(xy_out.div(self.temperature), dim=1)
        xy_distance = xy_prob.div(self.temperature).exp()#.clamp(min=1e-5, max=torch.finfo(torch.float32).max)
        previous_size = xy_distance.size()
        xy_distance = xy_distance.view(-1,self.xy_distribution_size**2)#flatten batch_size, seq_len for multinomial
        predicted_xy_coords = torch.multinomial(xy_distance, 1)
        predicted_xy_coords = predicted_xy_coords.view(previous_size[:-1])#get back batch_size, seq_len dimension
        predicted_xy_coords = self.convert_to_coordinates(predicted_xy_coords)#from index numbers between [0, self.xy_distribution_size*self.xy_distribution_size] to x, y [0,1] pairs
        
        wh_out_input = torch.cat((decoder_emb, class_one_hot, embedded_wh), dim=2)
        predicted_whs = self.wh_out(wh_out_input)
        predicted_whs = self.wh_sigmoid(predicted_whs)
        #TODO what is decoder_emb used for later?
        #in decoderRNN instead of decoder_emb is hiddenRNN
        return class_prob, predicted_class, xy_prob, predicted_whs, decoder_emb, predicted_xy_coords

    def convert_to_coordinates(self, input_coordinates):
        """
        Function to convert the input coordinates to a x,y value.
        The input coordinate is a value between [0...., xy_distribution_size**2]. The output x,y values are values between 0 and 1.
        """
        number_of_sectors = self.xy_distribution_size

        # First obtain the coordinates of the matrix
        x, y = input_coordinates % number_of_sectors, input_coordinates.div(number_of_sectors,rounding_mode='trunc')

        # Obtain the [x,y] value in [0, 1] range
        x_value = x.true_divide(number_of_sectors)
        y_value = y.true_divide(number_of_sectors)
        return torch.stack((x_value, y_value), dim=2)

def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)
#quick test and example.
if __name__ == "__main__":
    decoder = DecoderTRAN({'index2word':[1,2,3]}, 512, 8, 8, True, dropout_p=0.2, xy_distribution_size=16)
    tgt_l = torch.rand(32,5).to(torch.int64)
    tgt_x = torch.rand(32,5)
    tgt_y = torch.rand(32,5)
    tgt_w = torch.rand(32,5)
    tgt_h = torch.rand(32,5)
    encoder_output = torch.rand(32,15,512)
    predicted_classes, xy_prob, predicted_whs, decoder_emb, predicted_xy_coords = decoder(tgt_l,tgt_x, tgt_y, tgt_w, tgt_h,encoder_output,encoder_output_mask)
    print("final class size")
    print(predicted_classes.size())
    print("final xy size")
    print(predicted_xy_coords.size())
    print("decoder embeddings")
    print(decoder_emb.size())