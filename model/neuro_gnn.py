"""
Code from https://github.com/USC-InfoLab/NeuroGNN/blob/main/model/model.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch import Tensor
from model.eeg_transformer import EEGTranformer
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from model.neuro_constants import CORTEX_REGIONS_DESCRIPTIONS, ELECTRODES_BROADMANN_MAPPING, BROADMANN_AREA_DESCRIPTIONS, INCLUDED_CHANNELS, META_NODE_INDICES, ELECTRODES_REGIONS
from torch.optim.lr_scheduler import CosineAnnealingLR

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2), dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
    
class NeuroGNN_GNN_GCN(nn.Module):
    def __init__(self, node_feature_dim,
                 device='cpu', conv_hidden_dim=64, conv_num_layers=3):
        super(NeuroGNN_GNN_GCN, self).__init__()
        self.node_feature_dim = node_feature_dim      
        self.conv_hidden_dim = conv_hidden_dim
        self.conv_layers_num = conv_num_layers

        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False)) 
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.conv_hidden_dim) for _ in range(self.conv_layers_num-1)
        )
        for l in range(self.conv_layers_num-1):
            self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False))
        
        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        self.to(device)

    def forward(self, X, adj_mat):
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)
        X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        X_gnn = F.relu(X_gnn)
        for stack_i in range(1, self.conv_layers_num):
            X_res = X_gnn # Store the current state for the residual connection
            X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
            X_gnn = F.relu(self.layer_norms[stack_i-1](X_gnn + X_res)) # Add the residual connection
        return X_gnn

class NeuroGNN_Encoder(nn.Module):
    def __init__(self, input_dim, seq_length, nodes_num=19, meta_nodes_num=6,
                 semantic_embs=None, semantic_embs_dim=512,
                 dropout_rate=0.2, leaky_rate=0.2,
                 device='cpu', gru_dim=512, num_heads=8,
                 conv_hidden_dim=256, conv_num_layers=3,
                 output_dim=512,
                 dist_adj=None,
                 temporal_embs_dim=512,
                 gnn_block_type='gcn',
                 meta_node_indices=None):
        super(NeuroGNN_Encoder, self).__init__()

        self.graph_constructor = NeuroGNN_GraphConstructor(input_dim=input_dim, 
                                                           seq_length=seq_length, 
                                                           nodes_num=nodes_num, 
                                                           meta_nodes_num=meta_nodes_num,
                                                           semantic_embs=semantic_embs, 
                                                           semantic_embs_dim=semantic_embs_dim,
                                                           dropout_rate=dropout_rate, 
                                                           leaky_rate=leaky_rate,
                                                           device=device, 
                                                           gru_dim=gru_dim, 
                                                           num_heads=num_heads,
                                                           dist_adj=dist_adj,
                                                           temporal_embs_dim=temporal_embs_dim,
                                                           meta_node_indices=meta_node_indices)
        
        self.node_features_dim = temporal_embs_dim+semantic_embs_dim
        self.conv_hidden_dim = conv_hidden_dim
        self.output_dim = output_dim
        
        if gnn_block_type.lower() == 'gcn':
            # TODO: update conv_hidden_dim
            self.gnn_block = NeuroGNN_GNN_GCN(node_feature_dim=self.node_features_dim,
                                            device=device,
                                            conv_hidden_dim=conv_hidden_dim,
                                            conv_num_layers=conv_num_layers)
        else:
            raise ValueError("Not implemented")
        # elif gnn_block_type.lower() == 'stemgnn':
        #     self.gnn_block = NeuroGNN_StemGNN_Block(node_feature_dim=self.node_features_dim,
        #                                             device=device,
        #                                             output_dim=conv_hidden_dim,
        #                                             stack_cnt=2,
        #                                             conv_hidden_dim=conv_hidden_dim)
        # elif gnn_block_type.lower() == 'graphconv':
        #     self.gnn_block = NeuroGNN_GNN_GraphConv(node_feature_dim=self.node_features_dim,
        #                                             device=device,
        #                                             conv_hidden_dim=conv_hidden_dim,
        #                                             conv_num_layers=conv_num_layers)
        self.layer_norm = nn.LayerNorm(self.conv_hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(self.node_features_dim, self.conv_hidden_dim),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(int(self.conv_hidden_dim + self.node_features_dim), int(self.output_dim)),
            nn.ReLU()
        )
        self.to(device)

        
    def forward(self, x):
        X, adj_mat, (adj_mat_thresholded, adj_mat_unthresholded, embed_att, dist_adj, mhead_att_mat) = self.graph_constructor(x)
        X_gnn = self.gnn_block(X, adj_mat)
        # TODO: best way to make X_hat?
        # X_hat = torch.cat((X, X_gnn), dim=2)
        # X_hat = self.fc(X_hat)
        X = self.fc1(X)
        X_hat = self.layer_norm(X_gnn + X)
        return X_hat, adj_mat, (adj_mat_thresholded, adj_mat_unthresholded, embed_att, dist_adj, mhead_att_mat)

class NeuroGNN_GraphConstructor(nn.Module):
    def __init__(self, input_dim, seq_length, nodes_num=19, meta_nodes_num=6,
                 semantic_embs=None, semantic_embs_dim=256,
                 dropout_rate=0.0, leaky_rate=0.2,
                 device='cpu', gru_dim=256, num_heads=8,
                 dist_adj=None, temporal_embs_dim=256, meta_node_indices=None):
        super(NeuroGNN_GraphConstructor, self).__init__()
        self.gru_dim = gru_dim
        # self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.nodes_num = nodes_num
        self.meta_nodes_num = meta_nodes_num

        self.node_cluster_mapping = meta_node_indices + [list(range(nodes_num, nodes_num+meta_nodes_num))]
        self.seq1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(self.gru_dim*2)
        self.layer_norm2 = nn.LayerNorm(512)
        self.time_attention = Attention(self.gru_dim, self.gru_dim)
        self.mhead_attention = nn.MultiheadAttention(self.gru_dim*2, num_heads, dropout_rate, device=device, batch_first=True)
        # self.GRU_cells = nn.ModuleList(
        #     nn.GRU(512, gru_dim, batch_first=True, bidirectional=True) for _ in range(self.nodes_num+self.meta_nodes_num)
        # )
        # self.GRU_cells = nn.ModuleList(
        #     BiGRUWithMultiHeadAttention(256, gru_dim, 4) for _ in range(self.nodes_num+self.meta_nodes_num)
        # )
        self.bigru = nn.GRU(512, gru_dim, batch_first=True, bidirectional=True)
        # self.bigru_layernorm = nn.LayerNorm(gru_dim * 2)
        # self.bigru = Attentional_BiGRU(512, gru_dim, 4)
        # self.biGRU_cells = nn.ModuleList(
        #     nn.GRU(512, gru_dim, batch_first=True, bidirectional=True) for _ in range(len(self.node_cluster_mapping))
        # )
            
        
        # self.fc_ta = nn.Linear(gru_dim, self.time_step) #TODO remove this
        self.fc_ta = nn.Linear(gru_dim*2, temporal_embs_dim)
        self.layer_norm3 = nn.LayerNorm(temporal_embs_dim)
        self.layer_norm_sem = nn.LayerNorm(semantic_embs_dim)
        
        # for i, cell in enumerate(self.GRU_cells):
        #     cell.flatten_parameters()
            

        self.semantic_embs = torch.from_numpy(semantic_embs).to(device).float()
        
        # self.linear_semantic_embs = nn.Sequential(
        #     nn.Linear(semantic_embs.shape[1], semantic_embs_dim),
        #     nn.ReLU()
        # )
        self.linear_semantic_embs = nn.Linear(self.semantic_embs.shape[1], semantic_embs_dim) 
        # self.semantic_embs_layer_norm = nn.LayerNorm(semantic_embs_dim)
                
       
        # self.node_feature_dim = time_step + semantic_embs_dim
        self.node_feature_dim = temporal_embs_dim + semantic_embs_dim
        
    

        if dist_adj is not None:
            self.dist_adj = dist_adj
            self.dist_adj = torch.from_numpy(self.dist_adj).to(device).float()
        
        self.att_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Dropout):
                m.p = dropout_rate
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                m.in_proj_bias.data.zero_()
                m.out_proj.bias.data.zero_()  

        self.to(device) 
        self.device = device
        print(f"Device GraphConstrutor {device}")


    # def latent_correlation_layer(self, x):
    #     # batch_size, _, node_cnt = x.shape
    #     batch_size, seq_len, node_cnt, input_dim = x.shape
    #     # (node_cnt, batch_size, seq_len, input_dim)
    #     new_x = x.permute(2, 0, 1, 3)
    #     weighted_res = torch.empty(batch_size, node_cnt, self.gru_dim*2).to(x.get_device())
    #     for i, cell in enumerate(self.GRU_cells):
    #         # cell.flatten_parameters()
    #         cell.flatten_parameters()
    #         x_sup = self.seq1(new_x[i])
    #         gru_outputs, hid = cell(x_sup)
    #         out, hid = cell(x_sup)
    #         # TODO: multi-layer GRU?
    #         # hid = hid[-1, :, :]
    #         # hid = hid.squeeze(0)
    #         # gru_outputs = gru_outputs.permute(1, 0, 2).contiguous()
    #         # TODO: to or not to use self-attention?
    #         # weights = self.time_attention(hid, gru_outputs)
    #         # updated_weights = weights.unsqueeze(1)
    #         # gru_outputs = gru_outputs.permute(1, 0, 2)
    #         # weighted = torch.bmm(updated_weights, gru_outputs)
    #         # weighted = weighted.squeeze(1)
    #         # weighted_res[:, i, :] = self.layer_norm(weighted + hid)
    #         h_n = hid.permute(1, 0, 2)
    #         weighted_res[:, i, :] = h_n.reshape(batch_size, -1)
    #     _, attention = self.mhead_attention(weighted_res, weighted_res, weighted_res)

    #     attention = torch.mean(attention, dim=0) #[2000, 2000]
    #     # TODO: Should I put drop_out for attention?
    #     # attention = self.drop_out(attention)

    #     return attention, weighted_res
    
    # def latent_correlation_layer(self, x):
    #     # batch_size, _, node_cnt = x.shape
    #     batch_size, seq_len, node_cnt, input_dim = x.shape
    #     # (batch_size, node_cnt, seq_len, input_dim)
    #     new_x = x.permute(0, 2, 1, 3)
    #     weighted_res = torch.empty(batch_size, node_cnt, self.gru_dim*2).to(x.get_device())
    #     # get temporal contexts for nodes and meta-nodes
    #     for i, node_indices in enumerate(self.node_cluster_mapping):
    #         group_x = new_x[:, node_indices, :, :]
    #         group_x_reshaped = group_x.reshape(batch_size*len(node_indices), seq_len, input_dim)
    #         x_sup = self.seq1(group_x_reshaped)
    #         bigru_cell = self.biGRU_cells[i]
    #         bigru_cell.flatten_parameters()
    #         gru_outputs, hid = bigru_cell(x_sup)
    #         h_n = hid.permute(1, 0, 2)
    #         h_n_reshaped = h_n.reshape(batch_size, len(node_indices), -1)
    #         weighted_res[:, node_indices, :] = h_n_reshaped
    #     weighted_res = self.layer_norm(weighted_res)
    #     _, attention = self.mhead_attention(weighted_res, weighted_res, weighted_res)
    #     attention = torch.mean(attention, dim=0) #[2000, 2000]
    #     # TODO: Should I put drop_out for attention?
    #     # attention = self.drop_out(attention)

    #     return attention, weighted_res
    
    
    def latent_correlation_layer(self, x):
        batch_size, seq_len, node_cnt, input_dim = x.shape
        
        # Reshape x to combine the batch and node dimensions
        # New shape: (batch_size * node_cnt, seq_len, input_dim)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * node_cnt, seq_len, input_dim)

        # Pass x_reshaped through the desired layers (e.g., self.seq1 and the bigru)
        x_sup = self.seq1(x_reshaped)
        x_sup = self.layer_norm2(x_sup)
        self.bigru.flatten_parameters()
        gru_outputs, hid = self.bigru(x_sup)
        h_n = hid.permute(1, 0, 2)
        h_n_reshaped = h_n.reshape(batch_size, node_cnt, -1)
        
        # Apply Layer Normalization
        h_n_normalized = self.layer_norm(h_n_reshaped)

        _, attention = self.mhead_attention(h_n_normalized, h_n_normalized, h_n_normalized)

        attention = torch.mean(attention, dim=0) #[2000, 2000]

        return attention, h_n_normalized
    
    # def latent_correlation_layer(self, x):
    #     batch_size, seq_len, node_cnt, input_dim = x.shape
        
    #     # Reshape x to combine the batch and node dimensions
    #     # New shape: (batch_size * node_cnt, seq_len, input_dim)
    #     x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * node_cnt, seq_len, input_dim)

    #     # Pass x_reshaped through the desired layers (e.g., self.seq1 and the bigru)
    #     x_sup = self.seq1(x_reshaped)
    #     hid = self.bigru(x_sup)
    #     hid_reshaped = hid.reshape(batch_size, node_cnt, -1)

    #     _, attention = self.mhead_attention(hid_reshaped, hid_reshaped, hid_reshaped)

    #     attention = torch.mean(attention, dim=0) #[2000, 2000]

    #     return attention, hid_reshaped


    def forward(self, x):
        attention, weighted_res = self.latent_correlation_layer(x) 
        mhead_att_mat = attention.detach().clone()        
        
        weighted_res = self.fc_ta(weighted_res)
        weighted_res = F.relu(weighted_res)
        
        X = weighted_res.permute(0, 1, 2).contiguous()
        X = self.layer_norm3(X)
        if self.semantic_embs is not None:
            print(self.device)
            init_sem_embs = self.semantic_embs.to(self.device)
            transformed_embeds = self.linear_semantic_embs(init_sem_embs)
            # transformed_embeds = self.semantic_embs_layer_norm(transformed_embeds + init_sem_embs)
            # transformed_embeds = self.semantic_embs.to(x.get_device())
            transformed_embeds = self.layer_norm_sem(transformed_embeds)
            transformed_embeds = transformed_embeds.unsqueeze(0).repeat(X.shape[0], 1, 1)
            X = torch.cat((X, transformed_embeds), dim=2)
            
        embed_att = self.get_embed_att_mat_cosine(transformed_embeds)
        self.dist_adj = self.dist_adj.to(self.device)
        attention = ((self.att_alpha*self.dist_adj) + (1-self.att_alpha)*embed_att) * attention
        adj_mat_unthresholded = attention.detach().clone()
        
        
        attention_mask = self.case_amplf_mask(attention)
        
        attention[attention_mask==0] = 0
        adj_mat_thresholded = attention.detach().clone()
        

        # TODO: add softmax for attention?
        # attention = attention.softmax(dim=1)
        
        # X: Node features, attention: fused adjacency matrix
        return X, attention, (adj_mat_thresholded, adj_mat_unthresholded, embed_att, self.dist_adj, mhead_att_mat) 
        
    
    def _create_embedding_layers(self, embedding_size_dict, embedding_dim_dict, device):
        """construct the embedding layer, 1 per each categorical variable"""
        total_embedding_dim = 0
        cat_cols = list(embedding_size_dict.keys())
        embeddings = {}
        for col in cat_cols:
            embedding_size = embedding_size_dict[col]
            embedding_dim = embedding_dim_dict[col]
            total_embedding_dim += embedding_dim
            embeddings[col] = nn.Embedding(embedding_size, embedding_dim, device=device)
            
        return nn.ModuleDict(embeddings), total_embedding_dim
    
    
    def _normalize_attention(self, attention):
        # Normalize each row of the attention matrix
        max_scores, _ = torch.max(attention, dim=1, keepdim=True)
        norm_scores = attention / max_scores
        return norm_scores
    
    
    def case_amplf_mask(self, attention, p=2.5, threshold=0.08):
        '''
        This function computes the case amplification mask for a 2D attention tensor 
        with the given amplification factor p.

        Parameters:
            - attention (torch.Tensor): A 2D attention tensor of shape [n, n].
            - p (float): The case amplification factor (default: 2.5).
            - threshold (float): The threshold for the mask (default: 0.05).

        Returns:
            - mask (torch.Tensor): A 2D binary mask of the same size as `attention`,
              where 0s denote noisy elements and 1s denote clean elements.
        '''
        # Compute the maximum value in the attention tensor
        max_val, _ = torch.max(attention.detach(), dim=1, keepdim=True)

        # Compute the mask as per the case amplification formula
        mask = (attention.detach() / max_val) ** p

        # Turn the mask into a binary matrix, where anything below threshold will be considered as zero
        mask = torch.where(mask > threshold, torch.tensor(1).to(attention.device), torch.tensor(0).to(attention.device))
        return mask
        
        
    
    
    def get_embed_att_mat_cosine(self, embed_tensor):
        # embe_vecs: the tensor with shape (batch, POI_NUM, embed_dim)
        # Compute the dot product between all pairs of embeddings
        similarity_matrix = torch.bmm(embed_tensor, embed_tensor.transpose(1, 2))

        # Compute the magnitudes of each embedding vector
        magnitude = torch.norm(embed_tensor, p=2, dim=2, keepdim=True)

        # Normalize the dot product by the magnitudes
        normalized_similarity_matrix = similarity_matrix / (magnitude * magnitude.transpose(1, 2))

        # Apply a softmax function to obtain a probability distribution
        # similarity_matrix_prob = F.softmax(normalized_similarity_matrix, dim=2).mean(dim=0)
        
        return normalized_similarity_matrix.mean(dim=0).abs()
        # return similarity_matrix_prob
    
    
    def get_embed_att_mat_euc(self, embed_tensor):
        # Compute the Euclidean distance between all pairs of embeddings
        similarity_matrix = torch.cdist(embed_tensor[0], embed_tensor[0])

        # Convert the distances to similarities using a Gaussian kernel
        sigma = 1.0  # adjust this parameter to control the width of the kernel
        similarity_matrix = torch.exp(-similarity_matrix.pow(2) / (2 * sigma**2))

        # Normalize the similarity matrix by row
        # row_sum = similarity_matrix.sum(dim=1, keepdim=True)
        # similarity_matrix_prob = similarity_matrix / row_sum
        
        # return similarity_matrix
        return similarity_matrix

class NeuroGNN_Classification(nn.Module):
    def __init__(self, args, num_classes, device=None, dist_adj=None, initial_sem_embeds=None, metanodes_num=6, meta_node_indices=None):
        super(NeuroGNN_Classification, self).__init__()

        num_nodes = args.num_nodes
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim

        self.num_nodes = num_nodes
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes
        self.metanodes_num = metanodes_num
        
        self.meta_node_indices = meta_node_indices
        
        self.gnn_type = args.gnn_type

        self.encoder = NeuroGNN_Encoder(input_dim=enc_input_dim,
                                        seq_length=args.max_seq_len,
                                         output_dim=self.rnn_units,
                                        dist_adj=dist_adj,
                                        semantic_embs=initial_sem_embeds,
                                        gnn_block_type=self.gnn_type,
                                        meta_node_indices=self.meta_node_indices,
                                        device=device,
                                        )

        self.fc = nn.Linear(self.encoder.conv_hidden_dim, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()


    def forward(self, input_seq, seq_lengths=None, meta_node_indices=None):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            meta_node_indices: list of lists containing indices for each region
        Returns:
            logits: logits from last FC layer (before sigmoid/softmax)
        """
        node_embeds, _, _ = self.encoder(input_seq)
        pooled_embeddings = self.hierarchical_pooling(node_embeds, meta_node_indices=self.meta_node_indices)
        logits = self.fc(self.relu(self.dropout(pooled_embeddings)))

        return logits
    
    
    def hierarchical_pooling(self, node_embeddings, meta_node_indices):
        # Step 1: Pool Within Regions
        region_pooled_embeddings = [torch.max(node_embeddings[:, indices, :], dim=1)[0] for indices in meta_node_indices]
        region_pooled_embeddings = torch.stack(region_pooled_embeddings, dim=1) # Shape: (batch_size, num_regions, conv_dimension)

        # Step 2: Pool Across Meta Nodes
        meta_node_pooled_embeddings = torch.max(node_embeddings[:, -self.metanodes_num:, :], dim=1)[0] # Shape: (batch_size, conv_dimension)
        meta_node_pooled_embeddings = meta_node_pooled_embeddings.unsqueeze(1) # Add extra dimension, shape: (batch_size, 1, conv_dimension)
        
        # Step 3: Concatenate pooled embeddings
        all_pooled_embeddings = torch.cat([region_pooled_embeddings, meta_node_pooled_embeddings], dim=1) # Shape: (batch_size, num_regions + 1, conv_dimension)

        # Step 4: Max Pooling
        max_pooled_embeddings = torch.mean(all_pooled_embeddings, dim=1) # Shape: (batch_size, conv_dimension)

        return max_pooled_embeddings
    
def get_adjacency_matrix(distance_df, sensor_ids, dist_k=0.9):
    """
    Args:
        distance_df: data frame with three columns: [from, to, distance].
        sensor_ids: list of sensor ids.
        dist_k: threshold for graph sparsity
    Returns:
        adj_mx: adj
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.    
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[dist_mx > dist_k] = 0
   
    return adj_mx, sensor_id_to_ind, dist_mx

def get_meta_node_indices(electrodes_regions):
    meta_node_indices = defaultdict(list)

    for i, (node, region) in enumerate(electrodes_regions.items()):
        meta_node_indices[region].append(i)

    return dict(meta_node_indices)

def get_extended_adjacency_matrix(distance_df, sensor_ids, electrodes_regions, dist_k=0.9):
    adj_mat, sensor_id_to_ind, dist_mx = get_adjacency_matrix(distance_df, sensor_ids, dist_k)

    # map the sensor_id_to_ind to regions
    region_to_indices = get_meta_node_indices(electrodes_regions)

    # Get the number of regions and initialize a matrix for the meta nodes
    num_regions = len(region_to_indices)
    num_sensors = len(sensor_ids)
    meta_dist_mx = np.zeros((num_sensors + num_regions, num_sensors + num_regions))

    # Copy the original distance matrix to the extended matrix
    meta_dist_mx[:num_sensors, :num_sensors] = dist_mx

    # Calculate the mean distance for each region and add it to the matrix
    for region, indices in region_to_indices.items():
        region_index = num_sensors + list(region_to_indices.keys()).index(region)

        for i in range(num_sensors):
            meta_dist_mx[i, region_index] = dist_mx[i, indices].mean()
            meta_dist_mx[region_index, i] = dist_mx[indices, i].mean()

        for other_region, region_indices_j in region_to_indices.items():
            if other_region != region:
                other_region_index = num_sensors + list(region_to_indices.keys()).index(other_region)
                meta_dist_mx[region_index, other_region_index] = dist_mx[indices][:, region_indices_j].mean()
                meta_dist_mx[other_region_index, region_index] = dist_mx[region_indices_j][:, indices].mean()

    # Calculate the adjacency matrix using the Gaussian kernel and the threshold
    distances = meta_dist_mx[~np.isinf(meta_dist_mx)].flatten()
    std = distances.std()

    meta_adj_mat = np.exp(-np.square(meta_dist_mx / std)) #TODO WARNING HERE devision by zero!
    meta_adj_mat[meta_dist_mx > dist_k] = 0

    return meta_adj_mat, sensor_id_to_ind, meta_dist_mx

def get_electrode_descriptions(electrode_brodmann_map, brodmann_area_descrips):
    """map electrode names to brodmann areas descriptions and return a dictionary for electrode descriptions
    
    Args:
        electrode_brodmann_map (dict): electrode to brodmann area mapping
        brodmann_area_descrips (dict): brodmann area descriptions
            
    Returns:
        dict: electrode descriptions
    """
    electrode_descriptions = dict()
    for electrode, brodmann_area in electrode_brodmann_map.items():
        electrode_descriptions[electrode] = brodmann_area_descrips[brodmann_area]
    return electrode_descriptions

def get_semantic_embeds():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    llm = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    descriptions = []
    node_descriptions = get_electrode_descriptions(ELECTRODES_BROADMANN_MAPPING, BROADMANN_AREA_DESCRIPTIONS)
    for node, descp in node_descriptions.items():
        # descp = f'This node represents electrode {node.split()[1]} recordings. {descp}'
        descriptions.append(descp)
    for node, descp in CORTEX_REGIONS_DESCRIPTIONS.items():
        # descp = f'This is a meta-node that represents the recordings for {node} region of the cortext. {descp}'
        descriptions.append(descp)
    # global node description
    embeddings = llm.encode(descriptions)
    return embeddings

class NeuroGNN(EEGTranformer):
    def __init__(self, cfg, create_model=True):
        super().__init__(cfg, create_model=False)
        self.save_hyperparameters()
        # device = get_best_device()

        if create_model:       
            csv_path = Path(cfg.distance_csv_root) / cfg.distance_csv_path
            distances_df = pd.read_csv(csv_path)
            dist_adj, _, _ = get_extended_adjacency_matrix(distances_df, INCLUDED_CHANNELS, ELECTRODES_REGIONS)
            initial_sem_embs = get_semantic_embeds().to(self.device)
            # initial_sem_embs = initial_sem_embs[:19] # TODO Workaround to cut the regions
            
            self.model = NeuroGNN_Classification(cfg, 1, self.device, dist_adj, initial_sem_embs, meta_node_indices=META_NODE_INDICES)
        
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
    def forward(self, x):
        logits = self.model(x)
        return logits.squeeze(-1)
    
    def prediction(self, logits: Tensor):
        # 1) get probabilities
        probs = torch.sigmoid(logits)
        # 2) make binary predictions
        preds = (probs >= 0.5).long()
        return preds
    
    def loss_func(self, x: Tensor, y:Tensor):
        return self.criterion(x, y.float())
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.l2_wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.max_epochs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    