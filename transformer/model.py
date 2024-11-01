import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def clones(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def attention(query, key, value, mask=None, dropout=None):
    attention_weights = None
    attented_values = None
    # TODO: Implement attention mechanism
    # YOUR CODE STARTS HERE
    k_transpose = key.transpose(2,3)
    d_k = query.shape[3]
    q_k = torch.matmul(query, k_transpose)
    scaled_q_k = q_k * (1.0 / math.sqrt(d_k))
    if mask is not None:
        if (len(scaled_q_k.shape) != len(mask.shape)):
            mask = mask.unsqueeze(1)
        scaled_q_k = scaled_q_k.masked_fill(mask==0, float('-inf'))
    softmax = nn.Softmax(dim=-1)
    attention_weights = softmax(scaled_q_k)
    
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    attented_values = torch.matmul(attention_weights, value)
    # YOUR CODE ENDS HERE
    return attented_values, attention_weights


def autoregressive_mask(size):
    result = torch.tril(torch.ones((size, size), dtype=torch.bool)).unsqueeze(0)
    # YOUR CODE ENDS HERE
    return result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # TODO: Define the positional encoding class initialization
        # YOUR CODE STARTS HERE
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype = torch.float).unsqueeze(-1)
        denom =  torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos * denom)
        pe[:,1::2] = torch.cos(pos * denom)
        
        self.register_buffer("pe", pe)
        # YOUR CODE ENDS HERE

    def forward(self, x):
        # Prevent gradient computations for positional encodings
        out = None
        # TODO: Implement positional encoding forward pass
        # YOUR CODE STARTS HERE
        sequence_length = x.shape[1]
        x_add_pos = x + self.pe[:sequence_length, :]
        out = self.dropout(x_add_pos)
        # YOUR CODE ENDS HERE
        return out


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        Initializes the Embeddings module.
        Args:
            d_model (int): model dimensionality
            vocab_size (int): maximum number of tokens in the vocabulary
        """
        super(Embeddings, self).__init__()
        # TODO: Define embedding layer class initialization
        # YOUR CODE STARTS HERE
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        The forward pass of the Embeddings module.
        Converts input token indices into embeddings, scales the embeddings by the square root of the
        dimensionality of the model to maintain the variance.
        Parameters:
            x (Tensor): The input tensor of token indices. Expected shape [batch_size, sequence_length].

        Returns:
            Tensor: The scaled embeddings tensor. Shape [batch_size, sequence_length, d_model].
        """
        out = None
        # TODO: Implement embedding forward pass
        # YOUR CODE STARTS HERE
        out = self.embed(x) * math.sqrt(self.d_model)
        
        # YOUR CODE ENDS HERE
        return out 

    def set_embedding_weights(self):
        """
        Set the weights of the embedding layer.
        """
        for idx in range(self.vocab_size):
            self.embed.weight.data[idx] = torch.linspace(start=0.0, end=1.0, steps=self.d_model)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Initializes the MultiHeadedAttention module.
        Args:
            h (int): Number of attention heads.
            d_model (int): Total dimension of the model.
            dropout (float): Dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        # TODO: Define multi-headed attention class initialization
        # YOUR CODE STARTS HERE
        self.d_k = (d_model // h)
        self.h = h
        self.d_model = d_model
        
        self.q_linear = clones(nn.Linear(d_model, self.d_k, bias=True), h)
        self.k_linear = clones(nn.Linear(d_model, self.d_k,bias=True), h)
        self.v_linear = clones(nn.Linear(d_model, self.d_k,bias=True), h)
        self.projection_layer = nn.Linear(h * self.d_k, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        # YOUR CODE ENDS HERE

    def forward(self, query, key, value, mask=None):
        # TODO: Implement multi-headed attention forward pass
        # YOUR CODE STARTS HERE
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        _, seq_len_v, _ = key.shape
        q_projection = [linear(query) for linear in self.q_linear]
        v_projection =  [linear(value)for linear in self.v_linear]
        k_projection =  [linear(key) for linear in self.k_linear]
        q_projection = torch.cat(q_projection,dim=-1).view(batch_size, seq_len_q, -1, self.d_k).transpose(1,2)
        v_projection = torch.cat(v_projection,dim=-1).view(batch_size, seq_len_k, -1, self.d_k).transpose(1,2)
        k_projection = torch.cat(k_projection,dim=-1).view(batch_size, seq_len_v, -1, self.d_k).transpose(1,2)
   
        attented_values, self.attn, = attention(query=q_projection,key=k_projection, value=v_projection, mask=mask, dropout=self.dropout)
        
        attented_values = attented_values.transpose(1,2)
        attented_values = attented_values.contiguous().view(batch_size,-1,self.d_k * self.h)
        out = self.projection_layer(attented_values)
        # YOUR CODE ENDS HERE
        return out

    def set_weights(self, weights=None, biases=None):
        """
        Set the weights and biases of all the linear layers from external tensors.
        Parameters:
            weights (list of Tensors): A list of tensors for the weights of the linear layers.
            biases (list of Tensors): A list of tensors for the biases of the linear layers.
        """
        # TODO: Implement setting weights and biases for all layers
        # YOUR CODE STARTS HERE
        for i, l in enumerate(self.q_linear):
            l.weight.data.copy_(weights[0][i])
            l.bias.data.copy_(biases[0][i])
        for i, l in enumerate(self.k_linear):
            l.weight.data.copy_(weights[1][i])
            l.bias.data.copy_(biases[1][i])
        for i, l in enumerate(self.v_linear):
            l.weight.data.copy_(weights[2][i])
            l.bias.data.copy_(biases[2][i])
        self.projection_layer.weight.data.copy_(weights[3])
        self.projection_layer.bias.data.copy_(biases[3])
        # YOUR CODE ENDS HERE


class FeedForward(nn.Module):
        
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initializes the FeedForward network with specified dimensions and dropout.
        Args:
            d_model (int): The size of the input and output dimensions.
            d_ff (int): The dimensionality of the hidden layer.
            dropout (float): Dropout rate.
        """
        super(FeedForward, self).__init__()
        # TODO: Define the feedforward class initialization
        # YOUR CODE STARTS HERE
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(p= dropout)
        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        Defines the computation performed at every call of the feedforward network.
        Parameters:
            x (torch.Tensor): The input tensor with shape [batch_size, sequence_length, d_model].

        Returns:
            torch.Tensor: The output tensor with the same shape as input tensor, after being
                          processed by two linear layers and dropout.
        Apply dropout after the ReLU activation, which is added between w_1 and w_2.
        """
        out = None
        # TODO: Implement the feedforward forward pass
        # YOUR CODE STARTS HERE
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.w_2(x)
        # YOUR CODE ENDS HERE
        return out
    
    def set_weights(self, weights=None, biases=None):
        """
        Set the weights and biases of the linear layers from external tensors.
        Parameters:
            weights (list of Tensors): A list of tensors for the weights of the linear layers (0th index for first layer
            and 1st for second layer).
            biases (list of Tensors): A list of tensors for the biases of the linear layers. (0th index for first layer
            and 1st for second layer).
        """
        # TODO: Implement setting weights and biases for all layers
        # YOUR CODE STARTS HERE
        self.w_1.weight.data.copy_(weights[0])
        self.w_1.bias.data.copy_(biases[0])
        self.w_2.weight.data.copy_(weights[1])
        self.w_2.bias.data.copy_(biases[1])
        # YOUR CODE ENDS HERE


class LayerNorm(nn.Module):
    """
    Implements a Layer Normalization module as described in the cited literature.
    Attributes:
        scale_param (nn.Parameter): Scale parameter, learnable, initialized to ones.
        shift_param (nn.Parameter): Shift parameter, learnable, initialized to zeros.
        eps (float): A small constant added to the denominator for numerical stability.
    """
    def __init__(self, features, eps=1e-6):
        """
        Initializes the LayerNorm module with the specified number of features and a small
        epsilon value for numerical stability.
        Args:
            features (int): The number of individual features expected in the input.
            eps (float): A small constant to prevent any division by zero during normalization.
        """
        super(LayerNorm, self).__init__()
        # TODO: Define the layer normalization class initialization
        # YOUR CODE STARTS HERE
        self.scale_param = nn.Parameter(torch.ones(features))
        self.shift_param = nn.Parameter(torch.zeros(features))
        self.eps = eps
        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        Forward pass of the LayerNorm module.
        Parameters:
            x (torch.Tensor): Input tensor of shape [..., features].
        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        """
        out = None
        # TODO: Implement the layer normalization forward pass
        # YOUR CODE STARTS HERE
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        numerator = (x - mean)
        denom = std + self.eps
        fraction = numerator / denom
        out = self.scale_param * fraction+ self.shift_param
        # YOUR CODE ENDS HERE
        return out


class ResidualStreamBlock(nn.Module):
    def __init__(self, size, dropout):
        """
        Initializes the ResidualStreamBlock module with a specific size for normalization
        and a specified dropout rate.
        Args:
            size (int): The number of features in the input tensors expected by the layer normalization.
            dropout (float): The dropout probability to be used in the dropout layer.
        """
        super(ResidualStreamBlock, self).__init__()
        # TODO: Define the residual stream block class initialization
        # YOUR CODE STARTS HERE
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

        # YOUR CODE ENDS HERE

    def forward(self, x, sublayer):
        """
        Forward pass through the ResidualStreamBlock module which applies a residual connection
        followed by a dropout to the output of any sublayer function.
        This is for ease of use because this pattern is common in the transformer architecture.
        Apply dropout after the sublayer has been applied to the normalized input tensor, before skip connection.
        Parameters:
            x (torch.Tensor): The input tensor.
            sublayer (callable): A function or module that processes the normalized input tensor.
        Returns:
            torch.Tensor: The output tensor which is the element-wise addition of the input tensor
                          and the processed output from the sublayer, after dropout has been applied.
        """
        out = None
        # TODO: Implement the residual stream block forward pass
        # YOUR CODE STARTS HERE
        x_norm = self.norm(x)
        sublayer_out = sublayer(x_norm)
        out = self.dropout(sublayer_out) + x
        
        # YOUR CODE ENDS HERE
        return out


class EncoderBlock(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Initializes the EncoderLayer with self-attention and feed-forward network along with
        necessary configurations for residual stream blocks.
        Args:
            size (int): The size of the model (i.e., dimensionality of input and output).
            self_attn (nn.Module): An instance of a self-attention mechanism.
            feed_forward (nn.Module): An instance of a position-wise feed-forward network.
            dropout (float): Dropout rate for sublayers within the encoder.
        """
        super(EncoderBlock, self).__init__()
        # TODO: Define the encoder block class initialization
        # YOUR CODE STARTS HERE

        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual_stream_block1 = ResidualStreamBlock(size, dropout)
        self.residual_stream_block2 = ResidualStreamBlock(size, dropout)
        
        # YOUR CODE ENDS HERE

    def forward(self, x, mask):
        """
        Processes input through one encoder layer of a transformer model following the
        architecture specified in the original transformer paper.
        Don't forget the mask. It will handle the padding tokens, so they are not attended to.
        Parameters:
            x (torch.Tensor): The input tensor to the encoder layer.
            mask (torch.Tensor): The mask tensor to be applied during self-attention to
                                 prevent attention to certain positions.
        Returns:
            torch.Tensor: The output of the encoder layer after processing through self-attention
                          and feed-forward network with residual connections and normalization.
        """
        out = None
        # TODO: Implement the encoder block forward pass
        # YOUR CODE STARTS HERE
        x = self.residual_stream_block1(x, lambda x: self.self_attn(x, x, x,mask=mask))
        out = self.residual_stream_block2(x, lambda x: self.feed_forward(x))
        # YOUR CODE ENDS HERE
        return out


class Encoder(nn.Module):
    """
    Defines the core encoder which is a stack of N identical encoder blocks.
    Attributes:
        layers (nn.ModuleList): A list of identical layer modules that make up the encoder.
        norm (LayerNorm): A normalization layer applied to the output of the last encoder layer
                          to ensure that the output is normalized before it is passed to the
                          next stage of the model.
    """

    def __init__(self, layer, n_blocks):
        """
        Initializes the Encoder module with a stack of N identical layers and a final
        normalization layer.
        HINTS:
            Use the `clones` function to create N identical layers.
            The LayerNorm is applied to the output of the encoder stack. The layer size should be set accordingly.
            Can you use layer.size?
        Args:
            layer (nn.Module): The layer to be cloned and stacked.
            n_blocks (int): The number of times the layer should be cloned to form the encoder stack.
        """
        super(Encoder, self).__init__()
        # TODO: Define the encoder class initialization
        # YOUR CODE STARTS HERE
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

        # YOUR CODE ENDS HERE

    def forward(self, x, mask):
        """
        Processes the input sequence through each layer in the encoder block sequentially.
        HINT: you can use one loop to go through each layer in the encoder stack.
        Parameters:
            x (torch.Tensor): The input tensor to the encoder.
            mask (torch.Tensor): The mask tensor to be applied during the operations of each layer
                                 to prevent attention to certain positions, typically padding.

        Returns:
            torch.Tensor: The output of the encoder after all layers and the final normalization
                          have been applied.
        """
        out = None
        # TODO: Implement the encoder forward pass
        # YOUR CODE STARTS HERE
        for layer in self.layers:
            x =layer(x,mask)
        out = self.norm(x)
        # YOUR CODE ENDS HERE
        return out


class DecoderBlock(nn.Module):

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        """
        Initializes the DecoderLayer with specified self-attention, cross-attention,
        and feed-forward network along with necessary configurations for residual stream blocks.
        Args:
            size (int): The size of the model (i.e., dimensionality of input and output).
            self_attn (nn.Module): An instance of a self-attention mechanism.
            cross_attn (nn.Module): An instance of a cross-attention mechanism.
            feed_forward (nn.Module): An instance of a position-wise feed-forward network.
            dropout (float): Dropout rate for sublayers within the decoder.
        """
        super(DecoderBlock, self).__init__()
        # TODO: Define the decoder block class initialization
        # YOUR CODE STARTS HERE
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_foward = feed_forward
        self.residual_stream_block_1 = ResidualStreamBlock(size, dropout)
        self.residual_stream_block_2 = ResidualStreamBlock(size, dropout)
        self.residual_stream_block_3 = ResidualStreamBlock(size, dropout)
        
        # YOUR CODE ENDS HERE

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Processes input through one decoder layer of a transformer model following the architecture
        specified in Figure 1 (right) from the original transformer paper.
        Parameters:
            x (torch.Tensor): The input tensor to the decoder layer.
            memory (torch.Tensor): The output from the last layer of the encoder, serving as memory
                                   for the cross-attention mechanism.
            src_mask (torch.Tensor): The mask tensor for the encoder's output, used during cross-attention.
            tgt_mask (torch.Tensor): The mask tensor for the decoder's input, used during self-attention
                                     to prevent attending to subsequent positions.
        Returns:
            torch.Tensor: The output of the decoder layer after processing through self-attention,
                          cross-attention, and feed-forward network with residual connections and normalization.
        """
        out = None
        # TODO: Implement the decoder block forward pass
        # YOUR CODE STARTS HERE
        x = self.residual_stream_block_1(x, lambda x: self.self_attn(x, x, x,mask=tgt_mask))
        x = self.residual_stream_block_2(x, lambda x: self.cross_attn(x, memory, memory,mask=src_mask))
        out = self.residual_stream_block_3(x, self.feed_foward)

        # YOUR CODE ENDS HERE
        return out


class Decoder(nn.Module):

    def __init__(self, layer, n_blocks):
        """
        Initializes the Decoder with N identical layers and a layer normalization step
        at the end.
        Implement it similar to encoder.
        Args:
            layer (nn.Module): The decoder layer to be cloned and stacked.
            n_blocks (int): The number of layers in the decoder stack.
                              or output.
        """
        super(Decoder, self).__init__()
        # TODO: Define the decoder class initialization
        # YOUR CODE STARTS HERE
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        # YOUR CODE ENDS HERE

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Processes the input sequence through each decoder layer in sequence, using the
        output of the encoder as memory.
        Implement it similar to encoder, but remember that the decoder has memory from the encoder.
        Parameters:
            x (torch.Tensor): The input tensor to the decoder.
            memory (torch.Tensor): The output tensor from the encoder which serves as memory
                                   in cross-attention mechanisms.
            src_mask (torch.Tensor): The mask for the encoder output, used in cross-attention.
            tgt_mask (torch.Tensor): The mask for the decoder input, used in self-attention
                                     to prevent attention to subsequent positions.
        Returns:
            torch.Tensor: The output tensor from the decoder after passing through all layers
                          and normalization.
        """
        out = None
        # TODO: Implement the decoder forward pass
        # YOUR CODE STARTS HERE
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        out = self.norm(x)
        # YOUR CODE ENDS HERE
        return out


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        """
        Initializes the Generator module with a linear transformation.
        Args:
            d_model (int): The dimensionality of the input feature space.
            vocab (int): The size of the vocabulary for the output space.
        """
        super(Generator, self).__init__()
        # TODO: Define the generator class initialization
        # YOUR CODE STARTS HERE
        self.linear = nn.Linear(d_model, vocab, bias=True)
        # YOUR CODE ENDS HERE

    def forward(self, x):
        """
        Defines the forward pass of the Generator. Applies a linear transformation to the input
        tensor and then performs a log softmax on the result to produce a distribution over the
        vocabulary.
        Parameters:
            x (torch.Tensor): The input tensor containing features from the decoder.
        Returns:
            torch.Tensor: The log probability of each vocabulary token for each sequence in the batch.
        """
        out = None
        # TODO: Implement the generator forward pass
        # YOUR CODE STARTS HERE
        x = self.linear(x)
        x = x.softmax(dim=-1)
        out = torch.log(x)
        # YOUR CODE ENDS HERE
        return out
    
    def set_weights(self, weight=None, bias=None):
        """
        Set the weights and biases of the linear layer from external tensors.
        Parameters:
            weight (Tensor): A tensor for the weights of the linear layer.
            bias (Tensor): A tensor for the biases of the linear layer.
        """
        # TODO: Implement setting weights and biases for the linear layer
        # YOUR CODE STARTS HERE
        self.linear.weight.data.copy_(weight)
        self.linear.bias.data.copy_(bias)
        # YOUR CODE ENDS HERE


class Transformer(nn.Module):
    """
    Implements a standard Encoder-Decoder transformer. It
    combines an encoder and a decoder with embedding layers for the source and target
    sequences, and a final generator layer that typically produces probabilities over
    a target vocabulary.
    Attributes:
        encoder (nn.Module): The encoder module which processes the input sequence.
        decoder (nn.Module): The decoder module which generates the output sequence.
        src_embed (nn.Module): An embedding layer for the source sequence.
        tgt_embed (nn.Module): An embedding layer for the target sequence.
        generator (nn.Module): A generator layer that converts the output of the decoder
                               into a probability distribution over the target vocabulary.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        Initializes the EncoderDecoder model with its constituent components.
        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module.
            src_embed (nn.Module): Embedding layer for the source text.
            tgt_embed (nn.Module): Embedding layer for the target text.
            generator (nn.Module): Output generator layer.
        """
        super(Transformer, self).__init__()
        # TODO: Define the transformer class initialization
        # YOUR CODE STARTS HERE
        self.encoder = encoder
        self.decoder = decoder
        self.scr_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # YOUR CODE ENDS HERE

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Defines the forward pass of the Encoder-Decoder model using the provided source
        and target sequences along with their respective masks.
        Parameters:
            src (torch.Tensor): The source sequence input tensor.
            tgt (torch.Tensor): The target sequence input tensor.
            src_mask (torch.Tensor): The mask tensor for the source sequence.
            tgt_mask (torch.Tensor): The mask tensor for the target sequence.
        Returns:
            torch.Tensor: The output from the decoder which is then passed to the generator.
        """
        out = None
        # TODO: Implement the transformer forward pass
        # YOUR CODE STARTS HERE
        memory = self.encode(src, src_mask)
        
        out = self.decode(memory, src_mask,tgt, tgt_mask)
       # out = self.generator(decoder_output)

        # YOUR CODE ENDS HERE
        return out

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.
        Parameters:
            src (torch.Tensor): The source sequence tensor.
            src_mask (torch.Tensor): The mask tensor for the source sequence.
        Returns:
            torch.Tensor: The encoded output, which serves as the context for the decoder.
        """
        out = None
        # TODO: Implement the encoding function
        # YOUR CODE STARTS HERE
        out = self.encoder(self.scr_embed(src), mask=src_mask)
        # YOUR CODE ENDS HERE
        return out

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence using the encoded source as context.
        Parameters:
            memory (torch.Tensor): The output from the encoder.
            src_mask (torch.Tensor): The mask for the source sequence, used in the decoder.
            tgt (torch.Tensor): The target sequence tensor.
            tgt_mask (torch.Tensor): The mask tensor for the target sequence.
        Returns:
            torch.Tensor: The output from the decoder.
        """
        out = None
        # TODO: Implement the decoding function
        # YOUR CODE STARTS HERE
        tgt_embed = self.tgt_embed(tgt)
        out = self.decoder(tgt_embed, memory, src_mask, tgt_mask)
        # YOUR CODE ENDS HERE
        return out


def make_model(src_vocab, tgt_vocab, n_blocks=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Constructs a Transformer model using specified hyperparameters and initializes it.
    Parameters:
        src_vocab (int): Size of the source vocabulary.
        tgt_vocab (int): Size of the target vocabulary.
        n_blocks (int, optional): Number of blocks in both the encoder and decoder. Default is 6.
        d_model (int, optional): Dimensionality of the input embeddings. Default is 512.
        d_ff (int, optional): Dimensionality of the feed-forward layer. Default is 2048.
        h (int, optional): Number of attention heads. Default is 8.
        dropout (float, optional): Dropout rate. Default is 0.1.
    Returns:
        nn.Module: A Transformer model configured with the specified hyperparameters.
    The model construction includes multi-head attention mechanisms, feed-forward networks,
    positional encodings for inputs, and embeddings for both source and target vocabularies.
    All parameters are initialized using the Xavier uniform initialization, which is crucial
    for deep learning models as it helps in maintaining a level of variance that is neither
    too small nor too large.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderBlock(d_model, c(attn), c(ff), dropout), n_blocks),
        Decoder(DecoderBlock(d_model, c(attn), c(attn), c(ff), dropout), n_blocks),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Xavier uniform (also known as Glorot initialization).
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
