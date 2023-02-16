from models.encoder import *
from models.decoder import *
from models.attention import *

from models.Trans_atten import Transforme_Encoder

class LPR_model(nn.Module):

    '''

    add GPM between the encoder and decoder.

    head and inner are parameters of GPM, the head number and dim of FFN inner features

    '''

    def __init__(self, nc, nclass, imgW=96, imgH=32, K=8, isSeqModel=True, head=2, inner=256, isl2Norm=False):

        super(LPR_model, self).__init__()


        self.isSeqModel =isSeqModel

        self.model_dim = 64


        self.encoder = CNNEncoder_baseline_light3(nc)

        self.attention = Deformable_Attention(nc=64, K=K, downsample=4)

        self.decoder = FCDecoder(nclass, input_dim=64)


        # insert GPM between the encoder and decoder

        if self.isSeqModel:


            key_dim = int(self.model_dim/head)

            self.SequenceModeling = Transforme_Encoder(d_word_vec=self.model_dim, n_layers=2, n_head=head, d_k=key_dim, d_v=key_dim,

                d_model=self.model_dim, d_inner=inner, dropout=0.1, n_position=192)


        self.isl2Norm = isl2Norm


    def forward(self, input, isTsne=False, isVis=False, isSAVis=False, isCNN=False):

        
        conv_out,_ = self.encoder(input)


        if self.isSeqModel:

            b, c, h, w = conv_out.shape

            conv_out = conv_out.permute(0, 1, 3, 2)

            conv_out = conv_out.contiguous().view(b, c, -1)

            conv_out = conv_out.permute(0, 2, 1)

            enc_slf_attn, seq_out = self.SequenceModeling(conv_out, src_mask=None)

            seq_out = seq_out.permute(0, 2, 1).reshape(b, c, w, h).permute(0, 1, 3, 2)


            atten_list, atten_out = self.attention(seq_out)

            
        else:

            atten_list, atten_out = self.attention(conv_out)

        
        #l2 norm to attentioned features

        if self.isl2Norm:

            atten_out = F.normalize(atten_out, 2, dim=2)


        text_preds = self.decoder(atten_out)


        return atten_list, text_preds, enc_slf_attn