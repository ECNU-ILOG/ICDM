import wandb as wb
import numpy as np
import scipy.sparse as sp
import dgl
# from dgl.nn.pytorch import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from tqdm import tqdm
import warnings
from runners.commonutils.datautils import transform, get_doa_function, get_r_matrix, get_group_acc
from runners.commonutils.util import get_number_of_params, NoneNegClipper
from runners.ICDM.utils import l2_loss, dgl2tensor, concept_distill, get_subgraph
from dgl.base import DGLError
from dgl import function as fn
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.nn.pytorch import GATConv, GATv2Conv

# from dgl.nn.pytorch import SAGEConv

warnings.filterwarnings('ignore')


class SAGEConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=True,
            norm=None,
            activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                                             : graph.num_dst_nodes()
                                             ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                        degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = h_neigh
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = graph.dstdata["neigh"]
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = h_self + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class Attn(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attn, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class SAGENet(nn.Module):
    def __init__(self, dim, layers_num=2, type='mean', device='cpu', drop=True):
        super(SAGENet, self).__init__()
        self.drop = drop
        self.type = type
        self.layers = []
        for i in range(layers_num):
            if type == 'mean':
                self.layers.append(SAGEConv(in_feats=dim, out_feats=dim, aggregator_type=type).to(device))
            elif type == 'gat':
                self.layers.append(GATConv(in_feats=dim, out_feats=dim, num_heads=4).to(device))
            elif type == 'gatv2':
                self.layers.append(GATv2Conv(in_feats=dim, out_feats=dim, num_heads=4).to(device))

    def forward(self, g, h):
        outs = [h]
        tmp = h
        from dgl import DropEdge
        for index, layer in enumerate(self.layers):
            drop = DropEdge(p=0.05 + 0.1 * index)
            if self.drop:
                if self.training:
                    g = drop(g)
                if self.type != 'mean':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            else:
                if self.type != 'mean':
                    g = dgl.add_self_loop(g)
                    tmp = torch.mean(layer(g, tmp), dim=1)
                else:
                    tmp = layer(g, tmp)
            outs.append(tmp / (1 + index))
        res = torch.sum(torch.stack(
            outs, dim=1), dim=1)
        return res


class IGNet(nn.Module):
    def __init__(self, stu_num, prob_num, know_num, dim, graph, norm_adj=None, inter_layers=3, hidden_dim=512,
                 device='cuda',
                 gcnlayers=3, agg_type='mean', cdm_type='lightgcn', khop=2):
        super().__init__()
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.dim = dim
        self.graph = graph
        self.device = device
        self.norm_adj = norm_adj.to(self.device)
        self.cdm_type = cdm_type
        self.khop = khop

        self.stu_emb = nn.Embedding(self.stu_num, self.dim)
        self.exer_emb_right = nn.Embedding(self.prob_num, self.dim)
        self.exer_emb_wrong = nn.Embedding(self.prob_num, self.dim)
        self.know_emb = nn.Embedding(self.know_num, self.dim)
        self.global_user = nn.Parameter(torch.zeros(size=(1, self.dim)))
        self.gcn_layers = gcnlayers

        self.S_E_right = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)
        self.S_E_wrong = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)
        self.E_C_right = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)
        self.E_C_wrong = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)
        self.S_C = SAGENet(dim=self.dim, type=agg_type, device=device, layers_num=self.khop)

        self.attn_S = Attn(self.dim, attn_drop=0.2)
        self.attn_E_right = Attn(self.dim, attn_drop=0.2)
        self.attn_E_wrong = Attn(self.dim, attn_drop=0.2)
        self.attn_E = Attn(self.dim, attn_drop=0.2)
        self.attn_C = Attn(self.dim, attn_drop=0.2)

        self.Involve_Matrix = dgl2tensor(self.graph['I'])[:self.stu_num, self.stu_num:].to(self.device)
        self.transfer_stu_layer = nn.Linear(self.dim, self.know_num)
        self.transfer_exer_layer = nn.Linear(self.dim, self.know_num)
        self.transfer_concept_layer = nn.Linear(self.dim, self.know_num)

        self.change_latent_stu_mirt = nn.Linear(self.know_num, 16)
        self.change_latent_exer_mirt = nn.Linear(self.know_num, 16)

        self.change_latent_stu_irt = nn.Linear(self.know_num, 1)
        self.change_latent_exer_irt = nn.Linear(self.know_num, 1)

        self.disc_emb = nn.Embedding(self.prob_num, 1)
        layers = []
        for i in range(inter_layers):
            layers.append(nn.Linear(self.know_num if i == 0 else hidden_dim // pow(2, i - 1), hidden_dim // pow(2, i)))
            layers.append(nn.Dropout(p=0.3))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim // pow(2, inter_layers - 1), 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        BatchNorm_names = ['layers.{}.weight'.format(4 * i + 1) for i in range(3)]
        for index, (name, param) in enumerate(self.named_parameters()):
            if 'weight' in name:
                if name not in BatchNorm_names:
                    nn.init.xavier_normal_(param)

    def compute(self, emb):
        all_emb = emb
        embs = [emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        out_embs = torch.mean(torch.stack(embs, dim=1), dim=1)
        return out_embs

    def forward(self, stu_id, exer_id, knowledge_point):
        concept_id = torch.where(knowledge_point != 0)[1].to(self.device)
        concept_id_S = concept_id + torch.full(concept_id.shape, self.stu_num).to(self.device)
        concept_id_E = concept_id + torch.full(concept_id.shape, self.prob_num).to(self.device)
        exer_id_S = exer_id + torch.full(exer_id.shape, self.stu_num).to(self.device)

        subgraph_node_id_Q = torch.cat((exer_id.detach().cpu(), concept_id_E.detach().cpu()), dim=-1)
        subgraph_node_id_R = torch.cat((stu_id.detach().cpu(), exer_id_S.detach().cpu()), dim=-1)
        subgraph_node_id_I = torch.cat((stu_id.detach().cpu(), concept_id_S.detach().cpu()), dim=-1)

        R_subgraph_Right = get_subgraph(self.graph['right'], subgraph_node_id_R, device=self.device)
        R_subgraph_Wrong = get_subgraph(self.graph['wrong'], subgraph_node_id_R, device=self.device)
        I_subgraph = get_subgraph(self.graph['I'], subgraph_node_id_I, device=self.device)
        Q_subgraph = get_subgraph(self.graph['Q'], subgraph_node_id_Q, device=self.device)

        exer_info_right = self.exer_emb_right.weight
        exer_info_wrong = self.exer_emb_wrong.weight
        concept_info = self.know_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info]).to(self.device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(self.device)

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.stu_emb.weight
        S_C = torch.cat([stu_info, concept_info]).to(self.device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(self.device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(self.device)
        S_E_info_right = self.S_E_right(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.S_E_wrong(R_subgraph_Wrong, S_E_wrong)
        S_C_info = self.S_C(I_subgraph, S_C)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.prob_num], S_E_info_right[self.stu_num:]])
        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.prob_num], S_E_info_wrong[self.stu_num:]])
        E_forward = E_forward_right * E_forward_wrong
        C_forward = self.attn_C.forward(
            [E_C_info_right[self.prob_num:], E_C_info_wrong[self.prob_num:], S_C_info[self.stu_num:]])
        S_forward = self.attn_S.forward(
            [S_E_info_right[:self.stu_num], S_E_info_wrong[:self.stu_num], S_C_info[:self.stu_num]])

        emb = torch.cat([S_forward, E_forward]).to(self.device)
        disc = torch.sigmoid(self.disc_emb(exer_id))

        def irf(theta, a, b, D=1.702):
            return torch.sigmoid(torch.mean(D * a * (theta - b), dim=1)).to(self.device).view(-1)

        if self.cdm_type == 'lightgcn':
            out = self.compute(emb)
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                out[:self.stu_num]), self.transfer_exer_layer(out[self.stu_num:]), self.transfer_concept_layer(
                C_forward)
            exer_concept_distill = concept_distill(knowledge_point, C_forward)
            state = disc * (torch.sigmoid(S_forward[stu_id] * exer_concept_distill) - torch.sigmoid(
                E_forward[exer_id] * exer_concept_distill)) * knowledge_point
            return self.layers(state).view(-1)

        elif self.cdm_type == 'ncdm':
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(C_forward)
            state = disc * (torch.sigmoid(S_forward[stu_id]) - torch.sigmoid(
                E_forward[exer_id])) * knowledge_point
            return self.layers(state).view(-1)

        elif self.cdm_type == 'mirt':
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(C_forward)
            return irf(S_forward[stu_id], disc, E_forward[exer_id])

        # elif self.cdm_type == 'mirt':
        #     S_forward, E_forward, C_forward = self.transfer_stu_layer(
        #         S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(
        #         C_forward)
        #     return (1 / (1 + torch.exp(
        #         - torch.sum(torch.multiply(S_forward[stu_id] * knowledge_point, E_forward[exer_id] * knowledge_point),
        #                     dim=-1, keepdim=True) + disc))).view(-1)
        #
        elif self.cdm_type == 'irt':
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(
                C_forward)

            return irf(S_forward[stu_id], disc, E_forward[exer_id])

        else:
            raise ValueError('We do not support it yet')

    def get_mastery_level(self, pin_memory=False):
        if pin_memory:
            device = 'cpu'
        else:
            device = self.device
        R_subgraph_Right = self.graph['right'].to(device)
        R_subgraph_Wrong = self.graph['wrong'].to(device)
        I_subgraph = self.graph['I'].to(device)
        Q_subgraph = self.graph['Q'].to(device)

        exer_info_right = self.exer_emb_right.weight
        exer_info_wrong = self.exer_emb_wrong.weight
        concept_info = self.know_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info]).to(device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(device)

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.stu_emb.weight
        S_C = torch.cat([stu_info, concept_info]).to(device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(device)
        S_E_info_right = self.S_E_right(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.S_E_wrong(R_subgraph_Wrong, S_E_wrong)

        self.attn_S = self.attn_S.to(device)
        self.attn_C = self.attn_C.to(device)
        self.attn_E_right = self.attn_E_right.to(device)
        self.attn_E_wrong = self.attn_E_wrong.to(device)
        self.norm_adj = self.norm_adj.to(device)
        self.transfer_stu_layer = self.transfer_stu_layer.to(device)
        self.transfer_exer_layer = self.transfer_exer_layer.to(device)
        self.transfer_concept_layer = self.transfer_concept_layer.to(device)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.prob_num], S_E_info_right[self.stu_num:]])
        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.prob_num], S_E_info_wrong[self.stu_num:]])
        E_forward = E_forward_right * E_forward_wrong

        S_C_info = self.S_C(I_subgraph, S_C)
        C_forward = self.attn_C.forward(
            [E_C_info_right[self.prob_num:], E_C_info_wrong[self.prob_num:], S_C_info[self.stu_num:]])
        S_forward = self.attn_S.forward(
            [S_E_info_right[:self.stu_num], S_E_info_wrong[:self.stu_num], S_C_info[:self.stu_num]])

        if self.cdm_type == 'lightgcn':
            emb = torch.cat([S_forward, E_forward]).to(device)
            out = self.compute(emb)
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                out[:self.stu_num]), self.transfer_exer_layer(out[self.stu_num:]), self.transfer_concept_layer(
                C_forward)
            stu_concept_distill = concept_distill(self.Involve_Matrix, C_forward)
            return torch.sigmoid(S_forward * stu_concept_distill).detach().cpu().numpy()
        else:
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(
                C_forward)
            return torch.sigmoid(S_forward).detach().cpu().numpy()

    def apply_clipper(self):
        clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(clipper)


class ICDM(CDM):
    def __init__(self, stu_num, prob_num, know_num, dim=64, device='cuda:0', graph=None, gcn_layers=3, agg_type='mean',
                 weight_reg=0.05, wandb=True, cdm_type='lightgcn', khop=2, exist_idx=None, new_idx=None):
        super(ICDM, self).__init__()
        self.net = None
        self.know_num = know_num
        self.prob_num = prob_num
        self.stu_num = stu_num
        self.device = device
        self.dim = dim
        self.wandb = wandb
        self.agg_type = agg_type
        self.graph = graph
        self.gcn_layers = gcn_layers
        self.weight_reg = weight_reg
        self.cdm_type = cdm_type
        self.khop = khop
        self.mas_list = []
        self.exist_idx = exist_idx
        self.new_idx = new_idx

    def train(self, np_train, np_test, np_test_new=None, q=None, batch_size=None, epoch=10, lr=0.0005):
        train_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)
            for _ in [np_train, np_test]
        ]

        def get_adj_matrix(tmp_adj):
            adj_mat = tmp_adj + tmp_adj.T
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            return adj_matrix

        def sp_mat_to_sp_tensor(sp_mat):
            coo = sp_mat.tocoo().astype(np.float64)
            indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
            return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

        def create_adj_mat():
            n_nodes = self.stu_num + self.prob_num
            train_stu = np_train[:, 0]
            train_exer = np_train[:, 1]
            ratings = np.ones_like(train_stu, dtype=np.float64)
            tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.stu_num)), shape=(n_nodes, n_nodes))
            return sp_mat_to_sp_tensor(get_adj_matrix(tmp_adj))

        norm_adj = create_adj_mat()
        self.net = IGNet(stu_num=self.stu_num, prob_num=self.prob_num, know_num=self.know_num, dim=self.dim,
                         device=self.device, graph=self.graph, norm_adj=norm_adj, gcnlayers=self.gcn_layers,
                         agg_type=self.agg_type, cdm_type=self.cdm_type, khop=self.khop).to(self.device)
        r = get_r_matrix(np_test, self.stu_num, self.prob_num)
        if self.new_idx is not None:
            r_new = get_r_matrix(np_test_new, len(self.new_idx), self.prob_num, self.new_idx)
        bce_loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        get_number_of_params('igcdm', self.net)
        for epoch_i in range(epoch):
            epoch_losses = []
            bce_losses = []
            reg_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                stu_id, exer_id, knowledge_emb, y = batch_data
                stu_id: torch.Tensor = stu_id.to(self.device)
                exer_id: torch.Tensor = exer_id.to(self.device)
                knowledge_emb = knowledge_emb.to(self.device)
                y: torch.Tensor = y.to(self.device)
                tmp_E_right = self.net.exer_emb_right.weight
                tmp_E_wrong = self.net.exer_emb_wrong.weight
                reg_loss = l2_loss(
                    tmp_E_right[exer_id],
                    tmp_E_wrong[exer_id]
                )
                pred = self.net.forward(stu_id, exer_id, knowledge_emb)
                bce_loss = bce_loss_function(pred, y)
                bce_losses.append(bce_loss.mean().item())
                total_loss = bce_loss + self.weight_reg * reg_loss
                reg_losses.append((self.weight_reg * reg_loss).detach().cpu().numpy())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.net.apply_clipper()
                epoch_losses.append(total_loss.mean().item())
            print("[Epoch %d] average loss: %.6f, BCE loss: %.6f, reg loss: %.6f" % (
                epoch_i, float(np.mean(epoch_losses)), float(np.mean(bce_losses)), float(np.mean(reg_losses))))

            if self.wandb:
                if self.exist_idx is None:
                    auc, accuracy, rmse, f1, doa = self.eval(test_data, q=q,
                                                             r=r)
                    wb.define_metric("epoch")
                    wb.log({
                        'epoch': epoch_i,
                        'auc': auc,
                        'acc': accuracy,
                        'rmse': rmse,
                        'f1': f1,
                        'doa': doa,

                    })
                    print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f, DOA: %.6f" % (
                        epoch_i, auc, accuracy, rmse, f1, doa))
                else:
                    all_auc, all_acc, all_doa, new_auc, new_acc, new_doa = self.eval(test_data, q=q,
                                                                                     r=r, r_new=r_new)
                    wb.define_metric("epoch")
                    wb.log({
                        'epoch': epoch_i,
                        'all_auc': all_auc,
                        'all_acc': all_acc,
                        'all_doa': all_doa,
                        'new_auc': new_auc,
                        'new_acc': new_acc,
                        'new_doa': new_doa,

                    })
                    print("[Epoch %d] all_auc: %.6f, all_acc: %.6f all_DOA: %.6f" % (
                        epoch_i, all_auc, all_acc, all_doa))
                    print("[Epoch %d] new_auc: %.6f, new_acc: %.6f new_DOA: %.6f" % (
                        epoch_i, new_auc, new_acc, new_doa))

    def eval(self, test_data, q=None, r=None, r_new=None):
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        y_true_new, y_pred_new = [], []
        mas = self.net.get_mastery_level()
        self.mas_list.append(mas)
        doa_func = get_doa_function(know_num=self.know_num)
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, know_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(self.device)
            item_id: torch.Tensor = item_id.to(self.device)
            know_emb = know_emb.to(self.device)
            pred: torch.Tensor = self.net.forward(user_id, item_id, knowledge_point=know_emb)
            if self.new_idx is not None:
                for index, user in enumerate(user_id.detach().cpu().tolist()):
                    if user in self.new_idx:
                        y_true_new.append(y.tolist()[index])
                        y_pred_new.append(pred.detach().cpu().tolist()[index])
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        if self.exist_idx is None:
            return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,
                                                                  ), doa_func(mas, q.detach().cpu().numpy(),
                                                                              r)
        else:
            return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
                doa_func(mas, q.detach().cpu().numpy(), r), roc_auc_score(y_true_new, y_pred_new), accuracy_score(
                y_true_new, np.array(y_pred_new) >= 0.5), doa_func(mas[self.new_idx], q.detach().cpu().numpy(), r_new)
