import torch
import torch.nn as nn
from torch.nn import functional as F
from model.cell import DCGRUCell
import numpy as np
from .model import GNNPredictor, ConvKRegion, Embed2GraphByLinear, GruKRegion, Embed2GraphByProduct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class GCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.project1 = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.BatchNorm1d(roi_num),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(negative_slope=0.33)
        )

        self.project2 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.BatchNorm1d(roi_num),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(negative_slope=0.33)
        )

        self.project3 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.BatchNorm1d(roi_num),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(negative_slope=0.33)
        )

        self.fcn = nn.Sequential(
            nn.Linear(inner_dim, 32),
            nn.LeakyReLU(negative_slope=0.33),
            nn.Linear(32, 2)
        )

    def normalize(self, m):

        left = torch.sum(m, dim=2, keepdim=True)
        right = torch.sum(m, dim=1, keepdim=True)
        normalize = 1.0/torch.sqrt(torch.bmm(left, right))
        normalize[torch.isinf(normalize)] = 0
        return torch.mul(m, normalize)
        


    def forward(self, m, node_feature):

        m = self.normalize(m)

        x = self.project1(node_feature)

        x = torch.bmm(m, node_feature)

        x = self.project2(node_feature)

        x = torch.bmm(m, node_feature)

        x = self.project3(node_feature)

        x = torch.sum(x, dim=1)

        return self.fcn(x)



class Seq2SeqAttrs:
    def __init__(self, num_nodes=360):
        self.max_diffusion_step = 2
        self.cl_decay_steps = 1000
        self.filter_type = 'laplacian'
        self.num_nodes = num_nodes
        self.num_rnn_layers = 3
        self.rnn_units = 1
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, seq_len, input_dim=1, num_nodes=360):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, num_nodes=num_nodes)
        self.input_dim = input_dim
        self.seq_len = seq_len  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(
                output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # runs in O(num_layers) so not too slow
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, horizn=32, num_nodes=360):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, num_nodes=num_nodes)
        self.output_dim = 1
        self.horizn = horizn  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(
                output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)



class TSConstruction(nn.Module, Seq2SeqAttrs):
    def __init__(self, feature_dim=8, seq_len=64, node_num=360, discrete=True):
        super().__init__()
        Seq2SeqAttrs.__init__(self, num_nodes=node_num)
        self.seq_len = seq_len
        self.horizn_len = seq_len
        self.encoder_model = EncoderModel(seq_len, num_nodes=self.num_nodes)
        self.decoder_model = DecoderModel(seq_len, num_nodes=self.num_nodes)

        self.discrete = discrete

        self.extactor = GruKRegion(out_size=feature_dim)

        # self.graph_generator = Embed2GraphByLinear(
        #     input_dim=feature_dim, roi_num=self.num_nodes)

        self.graph_generator = Embed2GraphByProduct(
            input_dim=feature_dim, roi_num=self.num_nodes)

   

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)

        :return: encoder_hidden_state: ( batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            last_hidden_state, encoder_hidden_state = self.encoder_model(
                inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizn):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)

        outputs = torch.stack(outputs)
        return outputs

    def calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).to(device)
        d = torch.sum(adj_mx, 2)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(
            d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, full_seq, reconstruct_seq, node_feas, temperature):
        """
        :param inputs: shape (batch_size, num_sensor, seq_len)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        extracted_feature = self.extactor(full_seq)
        # if torch.any(torch.isnan(extracted_feature)):
        #     print('has nan1')
        # extracted_feature = F.softmax(extracted_feature, dim=-1)
        # if torch.any(torch.isnan(extracted_feature)):
        #     print('has nan2')

        adj = self.graph_generator(extracted_feature)
        if self.discrete:
            adj = gumbel_softmax(
                adj[:, :, :, 0], temperature=temperature, hard=True)
        else:
            adj = adj[:, :, :, 0]

        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
        adj = torch.where(mask, torch.zeros(
            mask.shape).to(device), adj)

        random_walk_matrix = self.calculate_random_walk_matrix(adj)

        random_walk_matrix = adj

        reconstruct_seq = reconstruct_seq.permute(2,0,1)

        encoder_hidden_state = self.encoder(reconstruct_seq, random_walk_matrix)
        outputs = self.decoder(encoder_hidden_state, random_walk_matrix)


        outputs = outputs.permute(1,2,0)

  

        return outputs


class BrainGSLModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, feature_dim=8, seq_len=64, node_num=360, discrete=True):
        super().__init__()
        Seq2SeqAttrs.__init__(self, num_nodes=node_num)
        self.seq_len = seq_len
        self.horizn_len = seq_len
        self.encoder_model = EncoderModel(seq_len, num_nodes=self.num_nodes)
        self.decoder_model = DecoderModel(seq_len, num_nodes=self.num_nodes)

        self.discrete = discrete

        self.extactor = GruKRegion(out_size=feature_dim)

        # self.graph_generator = Embed2GraphByLinear(
        #     input_dim=feature_dim, roi_num=self.num_nodes)

        self.graph_generator = Embed2GraphByProduct(
            input_dim=feature_dim, roi_num=self.num_nodes)

        self.predictor = GCNPredictor(
            node_input_dim=self.num_nodes, roi_num=self.num_nodes)

    def encoder(self, inputs, adj):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)

        :return: encoder_hidden_state: ( batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            last_hidden_state, encoder_hidden_state = self.encoder_model(
                inputs[t], adj, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizn):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)

        outputs = torch.stack(outputs)
        return outputs

    def calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).to(device)
        d = torch.sum(adj_mx, 2)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(
            d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, full_seq, reconstruct_seq, node_feas, temperature):
        """
        :param inputs: shape (batch_size, num_sensor, seq_len)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """

        extracted_feature = self.extactor(full_seq)
        # if torch.any(torch.isnan(extracted_feature)):
        #     print('has nan1')
        # extracted_feature = F.softmax(extracted_feature, dim=-1)
        # if torch.any(torch.isnan(extracted_feature)):
        #     print('has nan2')

        adj = self.graph_generator(extracted_feature)
        if self.discrete:
            adj = gumbel_softmax(
                adj[:, :, :, 0], temperature=temperature, hard=True)
        else:
            adj = adj[:, :, :, 0]

        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
        adj = torch.where(mask, torch.zeros(
            mask.shape).to(device), adj)

        random_walk_matrix = self.calculate_random_walk_matrix(adj)

        random_walk_matrix = adj

        reconstruct_seq = reconstruct_seq.permute(2,0,1)

        encoder_hidden_state = self.encoder(reconstruct_seq, random_walk_matrix)
        outputs = self.decoder(encoder_hidden_state, random_walk_matrix)


        outputs = outputs.permute(1,2,0)

        adj = torch.where(mask, torch.ones(
            mask.shape).to(device), adj)

        prediction = self.predictor(adj, node_feas)

        return outputs, prediction, adj
