import model
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PART 2.0
# Define the simple decoder model here with correct inputs as defined in models.py
def define_simple_decoder(hidden_size, input_vocab_len, output_vocab_len, max_length):
    """ Provides a simple decoder instance
        NOTE: Not all the function arguments are needed - you need to figure out which arguments to use

    :param hidden_size:
    :param input_vocab_len
    :param output_vocab_len
    :param max_length

    :return: a simple decoder instance
    """
    decoder = None

    # Write your implementation here
    decoder = model.DecoderRNN(hidden_size, output_vocab_len)
    # End of implementation

    return decoder


# PART 2.0
# Run the decoder model with correct inputs as defined in models.py
def run_simple_decoder(simple_decoder, decoder_input, encoder_hidden, decoder_hidden, encoder_outputs):
    """ Runs the simple_decoder
        NOTE: Not all the function arguments are needed - you need to figure out which arguments to use

    :param simple_decoder: the simple decoder object
    :param decoder_input:
    :param decoder_hidden:
    :param encoder_hidden:
    :param encoder_outputs:

    :return: The appropriate values
            HINT: Look at what the caller of this function in seq2seq.py expects as well as the simple decoder
                    definition in model.py
    """
    results = None

    # Write your implementation here
    results = simple_decoder.forward(decoder_input, decoder_hidden)
    # End of implementation

    return results  # Shape should be


# PART 2.1
class BidirectionalEncoderRNN(nn.Module):
    """Write class definition for BidirectionalEncoderRNN
    """

    def __init__(self, input_size, hidden_size):
        """

        :param input_size:
        :param hidden_size:
        """

        super(BidirectionalEncoderRNN, self).__init__()

        # Write your implementation here
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        # End of implementation

    def forward(self, input, hidden):
        """

        :param input:
        :param hidden:

        :return: output, hidden

            Hint: Don't correct the dimensions of the return values at this stage. Function skeletons for doing this are provided later.
            Sanity check: Shape of "output" should be [1, 1, 256] and shape of "hidden" should be [2, 1, 128]
        """

        # Write your implementation here
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
        # End of implementation

    def initHidden(self):
        return torch.zeros(1*2, 1, self.hidden_size, device=device)


# PART 2.1
# Define the encoder model here
def define_bi_encoder(input_vocab_len, hidden_size):
    """ Defines bidirectional encoder RNN

    :param input_vocab_len:
    :param hidden_size:
    :return: bidirectional encoder RNN
    """

    encoder = None

    # Write your implementation here
    encoder = BidirectionalEncoderRNN(input_vocab_len, hidden_size)
    # End of implementation

    return encoder


# PART 2.1
# Correct the dimension of encoder output by adding the forward and backward representation
def fix_bi_encoder_output_dim(encoder_output, hidden_size):
    """

    :param encoder_output:
    :param hidden_size:
    :return: output

    Sanity check: Shape of "output" should be [1, 1, 128]
    """
    output = None

    # Write your implementation here
    output = (encoder_output[:, :, :hidden_size] + encoder_output[:, :, hidden_size:])
    # End of implementation

    return output


# PART 2.1
# Correct the dimension of encoder hidden by considering only one sided layer
def fix_bi_encoder_hidden_dim(encoder_hidden):
    """

    :param encoder_hidden:
    :return: output

    Sanity check: Shape of "output" should be [1, 1, 128]
    """

    output = None

    # Write your implementation here
    output = encoder_hidden[-1:]
    # End of implementation

    return output



#Part 2.2
# PART 2.2.1
class AttnDecoderRNNDot(nn.Module):
    """
    Write class definition for AttnDecoderRNNDot
    Hint: Modify AttnDecoderRNN to use dot attention
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNDot, self).__init__()

        # Write your implementation here
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # End of implementation

    def forward(self, input, hidden, encoder_outputs):
        """
        Sanity check: Shape of "attn_weights" should be [1, 10]
        """
        # Write your implementation here
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(torch.matmul(hidden[0], torch.t(encoder_outputs)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        # End of implementation

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



# PART 2.2.2
class AttnDecoderRNNAdditive(nn.Module):
    """ 
    Write class definition for AttnDecoderRNNAdditive
    Hint: Modify AttnDecoderRNN to use additive attention
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNAdditive, self).__init__()

        # Write your implementation here
        self.max_length = max_length
        self.embed=nn.Embedding(output_size,hidden_size)
        self.dropout= nn.Dropout(dropout_p)
        self.gru= nn.GRU(hidden_size,hidden_size,dropout=dropout_p)
        self.tanh = nn.Tanh()
        self.value = nn.Linear(hidden_size,1)
        self.query= nn.Linear(2*hidden_size,hidden_size)
        self.attn_combine = nn.Linear(2*hidden_size, output_size)        
        # End of implementation

    def forward(self, input, hidden, encoder_outputs):
        """
        Sanity check: Shape of "attn_weights" should be [1, 10]
        """
        # Write your implementation here
        embed = self.embed(input)
        embed = embed.view(1,1,-1)
        output, hidden = self.gru(embed,hidden)
        hidden_rep = hidden[0].repeat(self.max_length,1)
        
        additive= self.query(torch.cat([hidden_rep,encoder_outputs],dim=-1))
        additive=self.tanh(additive)
        att_scores = self.value(additive)
        attn_weight = F.softmax(att_scores.T,dim=1)
        a=attn_weight@encoder_outputs
        
        a.unsqueeze_(dim=0)
        out= self.attn_combine(torch.cat([a,hidden],dim=-1)[0])
        output = F.log_softmax(out,dim=1)
        return output, hidden, attn_weight
        

        
        # End of implementation

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

