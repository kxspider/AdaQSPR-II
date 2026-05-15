# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import pathlib
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.utils.checkpoint import checkpoint

from ..config import MODEL_CONFIG_V2
from ..utils import logger, pad_1d_tokens, pad_2d, pad_coords
from ..weights import WEIGHT_DIR, weight_download_v2
from .transformersv2 import (AtomFeature, EdgeFeature, MovementPredictionHead,
                             SE3InvariantKernel, TransformerEncoderWithPairV2)
from torch_geometric.nn import GATv2Conv  # 
from torch_geometric.utils import to_undirected


BACKBONE = {
    'transformer': TransformerEncoderWithPairV2,
}

class UniMolV2Model(nn.Module):
    """
    UniMolModel is a specialized model for molecular, protein, crystal, or MOF (Metal-Organic Frameworks) data.
    It dynamically configures its architecture based on the type of data it is intended to work with. The model
    supports multiple data types and incorporates various architecture configurations and pretrained weights.

    Attributes:
        - output_dim: The dimension of the output layer.
        - data_type: The type of data the model is designed to handle.
        - remove_hs: Flag to indicate whether hydrogen atoms are removed in molecular data.
        - pretrain_path: Path to the pretrained model weights.
        - dictionary: The dictionary object used for tokenization and encoding.
        - mask_idx: Index of the mask token in the dictionary.
        - padding_idx: Index of the padding token in the dictionary.
        - embed_tokens: Embedding layer for token embeddings.
        - encoder: Transformer encoder backbone of the model.
        - gbf_proj, gbf: Layers for Gaussian basis functions or numerical embeddings.
        - classification_head: The final classification head of the model.
    """

    """
        Modify the model to support processing an input list consisting of 15 dictionaries, and construct a graph structure
    """

    def __init__(self, output_dim=1, model_size='84m', **params):
        """
        Initializes the UniMolModel with specified parameters and data type.

        :param output_dim: (int) The number of output dimensions (classes).
        :param data_type: (str) The type of data (e.g., 'molecule', 'protein').
        :param params: Additional parameters for model configuration.
        """
        super().__init__()

        self.args = molecule_architecture(model_size=model_size)
        self.output_dim = output_dim
        self.model_size = model_size
        self.remove_hs = params.get('remove_hs', False)

        name = model_size
        if not os.path.exists(
            os.path.join(WEIGHT_DIR, MODEL_CONFIG_V2['weight'][name])
        ):
            weight_download_v2(MODEL_CONFIG_V2['weight'][name], WEIGHT_DIR)

        self.pretrain_path = os.path.join(WEIGHT_DIR, MODEL_CONFIG_V2['weight'][name])

        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = 127
        self.embed_tokens = nn.Embedding(
            self.token_num, self.args.encoder_embed_dim, self.padding_idx
        )

        self.encoder = BACKBONE[self.args.backbone](
            num_encoder_layers=self.args.num_encoder_layers,
            embedding_dim=self.args.encoder_embed_dim,
            pair_dim=self.args.pair_embed_dim,
            pair_hidden_dim=self.args.pair_hidden_dim,
            ffn_embedding_dim=self.args.ffn_embedding_dim,
            num_attention_heads=self.args.num_attention_heads,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            activation_fn=self.args.activation_fn,
            droppath_prob=self.args.droppath_prob,
            pair_dropout=self.args.pair_dropout,
        )

        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512

        K = 128
        n_edge_type = 1

        self.atom_feature = AtomFeature(
            num_atom=num_atom,
            num_degree=num_degree,
            hidden_dim=self.args.encoder_embed_dim,
        )

        self.edge_feature = EdgeFeature(
            pair_dim=self.args.pair_embed_dim,
            num_edge=num_edge,
            num_spatial=num_spatial,
        )

        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=self.args.pair_embed_dim,
            num_pair=num_pair,
            num_kernel=K,
            std_width=self.args.gaussian_std_width,
            start=self.args.gaussian_mean_start,
            stop=self.args.gaussian_mean_stop,
        )

        self.movement_pred_head = MovementPredictionHead(
            self.args.encoder_embed_dim,
            self.args.pair_embed_dim,
            self.args.encoder_attention_heads,
        )

        self.classification_heads = nn.ModuleDict()
        self.dtype = torch.float32

        '''
        if 'pooler_dropout' in params:
            self.args.pooler_dropout = params['pooler_dropout']
        '''

        self.load_pretrained_weights(path=self.pretrain_path, strict=False)

        #dimensionality reduction CLS
        self.artificial_mlp0 = nn.Sequential(
            nn.Linear(self.args.encoder_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

        # Processing module for adding additional features
        self.artificial_mlp1 = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.artificial_mlp2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        # GAT
        self.gat1 = GATv2Conv(
            in_channels=64,
            out_channels=32,
            heads=2,
            edge_dim=64,  # edge feature dimension
            dropout=0.05  # dropout
        )

        self.norm1 = nn.LayerNorm(64)  # input 64，output 32*2=64

        self.gat2 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            edge_dim=64,  # edge feature dimension
            dropout=0.05  # dropout
        )
        self.norm2 = nn.LayerNorm(64)

        self.graph_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.output_dim)
        )


        # 
        '''
        self.regression_head = nn.Sequential(
            nn.Linear(self.args.encoder_embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.output_dim)
        )
        '''


    def load_pretrained_weights(self, path, strict=False):
        """
        Loads pretrained weights into the model.

        :param path: (str) Path to the pretrained weight file.
        """
        if path is not None:
            logger.info("Loading pretrained weights from {}".format(path))
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            try:
                weights = state_dict
                weights = {k: v for k, v in weights.items() if 'classification_head' not in k}
                self.load_state_dict(state_dict, strict=strict)
            except RuntimeError as e:
                if 'classification_head.dense.weight' in state_dict:
                    self.classification_head = ClassificationHead(
                        input_dim=self.args.encoder_embed_dim,
                        inner_dim=self.args.encoder_embed_dim,
                        num_classes=self.output_dim,
                        activation_fn=self.args.pooler_activation_fn,
                        pooler_dropout=self.args.pooler_dropout,
                    )
                    self.load_state_dict(state_dict, strict=strict)
                    logger.warning(
                        "This model is trained with the previous version. The classification_head is reset to previous version to load the model. This will be deprecated in the future. We recommend using the latest version of the model."
                    )
                else:
                    raise e

    @classmethod
    def build_model(cls, args):
        """
        Class method to build a new instance of the UniMolModel.

        :param args: Arguments for model configuration.
        :return: An instance of UniMolModel.
        """
        return cls(args)

    def _process_batch(self, sample_dict):
        """Process a sample to obtain cls_repr and artificial features"""
        # Extract necessary inputs
        atom_feat = sample_dict.get('atom_feat')
        atom_mask = sample_dict.get('atom_mask')
        edge_feat = sample_dict.get('edge_feat')
        shortest_path = sample_dict.get('shortest_path')
        degree = sample_dict.get('degree')
        pair_type = sample_dict.get('pair_type')
        attn_bias = sample_dict.get('attn_bias')
        src_tokens = sample_dict.get('src_tokens')
        src_coord = sample_dict.get('src_coord')
        artificial_feat = sample_dict.get('artificial_feat')

        #print(sample_dict['artificial_feat'].shape)

        # obtain cls_repr
        pos = src_coord
        n_mol, n_atom = atom_feat.shape[:2]
        token_feat = self.embed_tokens(src_tokens)
        x = self.atom_feature({'atom_feat': atom_feat, 'degree': degree}, token_feat)
        dtype = self.dtype
        x = x.type(self.dtype)

        n_mol, max_atom = atom_mask.shape[:2]

        # This is the main BIAS tensor for the transformer, including space for CLS.
        # It starts as zeros. It will be populated by edge_feature (2D) and se3_invariant_kernel (3D).
        graph_attn_bias = torch.zeros(
            n_mol, max_atom + 1, max_atom + 1, self.args.pair_embed_dim,
            device=x.device, dtype=self.dtype
        )
        
        # This populates the atom-specific parts of the bias tensor (from index 1 onwards)
        # with the 2D features like shortest_path.
        graph_attn_bias = self.edge_feature(
            {'shortest_path': shortest_path, 'edge_feat': edge_feat}, graph_attn_bias
        )
        
        # Now handle the attention MASK. This is different from the BIAS.
        # The input `attn_bias` tensor from the dataloader is actually the base for the mask.
        attn_mask = torch.zeros(
            n_mol, max_atom + 1, max_atom + 1, device=x.device, dtype=attn_bias.dtype
        )
        attn_mask[:, 1:, 1:] = attn_bias # attn_bias is the padded input from the dataloader
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.args.encoder_attention_heads, 1, 1
        )
        attn_mask = attn_mask.type(self.dtype)

        atom_mask_cls = torch.cat(
            [
                torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype),
                atom_mask,
            ],
            dim=1,
        ).type(self.dtype)

        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        def one_block(x, pos, return_x=False):
            delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = delta_pos.norm(dim=-1)
            attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
            new_attn_bias = graph_attn_bias.clone()
            new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d
            new_attn_bias = new_attn_bias.type(dtype)
            
            # Gradient Checkpointing for memory saving
            x, pair = checkpoint(
                self.encoder,
                x,
                new_attn_bias,
                atom_mask_cls,
                pair_mask,
                attn_mask,
                use_reentrant=False
            )

            node_output = self.movement_pred_head(
                x[:, 1:, :],
                pair[:, 1:, 1:, :],
                attn_mask[:, :, 1:, 1:],
                delta_pos.detach(),
            )
            if return_x:
                return x, pair, pos + node_output
            else:
                return pos + node_output
            #del delta_pos, dist, attn_bias_3d, new_attn_bias
            #torch.cuda.empty_cache()

        x, _, _ = one_block(x, pos, return_x=True)

        cls_repr = x[:, 0, :]  # 

        cls_repr_n = self.artificial_mlp0(cls_repr)
        # 
        artificial_hidden = self.artificial_mlp1(artificial_feat)
        #print(artificial_hidden.shape)
        #print(cls_repr.shape)
        combined_features = torch.cat([cls_repr_n, artificial_hidden], dim=-1)
        #print(combined_features.shape)
        node_feature = self.artificial_mlp2(combined_features)
        #print(node_feature.shape)

        return {
            'cls_repr': cls_repr_n,
            'node_feature': node_feature,
            'smiles_mark': sample_dict.get('smiles_mark', 0),
            'artificial_feat': artificial_feat
        }

    def build_graph_structure(self, processed_samples):
        #with torch.no_grad():  # 
        """
        Constructing a graph structure, including nodes, edges, node features, and edge features
        """
        batch_size = len(processed_samples)#8
        all_nodes = []
        all_edges = []
        all_node_features = []
        all_edge_features = []
        all_batch_indices = []

        for batch_idx in range(batch_size):
            # nodes
            #print(processed_samples[batch_idx])
            group1_samples = processed_samples[batch_idx][:3]
            group2_samples = processed_samples[batch_idx][3:6]

            # smiles_mark is not 'C'
            group1_nodes = [i for i, sample in enumerate(group1_samples)
                            if sample['smiles_mark'] == 1]
            group2_nodes = [i + 3 for i, sample in enumerate(group2_samples)
                            if sample['smiles_mark'] == 1]  # 

            # node features
            group1_features = [sample['node_feature'] for i, sample in enumerate(group1_samples)
                               if sample['smiles_mark'] == 1]
            group2_features = [sample['node_feature'] for i, sample in enumerate(group2_samples)
                               if sample['smiles_mark'] == 1]

            node_features = group1_features + group2_features
            n_nodes = len(node_features)

            if n_nodes == 0:
                continue  # 

            # Construct adjacency relationship: connect between groups, do not connect within groups
            edges = []
            for i in group1_nodes:
                for j in group2_nodes:
                    edges.append((i, j))

            # edge features were obtained from the last 9 samples
            edge_samples = processed_samples[batch_idx][6:15]
            edge_features = []
            edge_indices = []

            edge_idx_map = {}
            curr_edge_idx = 0
            for i, u in enumerate(group1_nodes):
                for j, v in enumerate(group2_nodes):
                    if u > 2 or v > 5:
                        raise ValueError("Node numbering is abnormal")
                    curr_edge_idx = u*3+v-3
                    edge_idx_map[(u, v)] = curr_edge_idx

            for edge_idx, (u, v) in enumerate(edges):
                if (u, v) in edge_idx_map:
                    sample_idx = edge_idx_map[(u, v)]
                    edge_sample = edge_samples[sample_idx]

                    if edge_sample['smiles_mark'] == 1:
                        edge_features.append(edge_sample['cls_repr'])
                        edge_indices.append([u, v])
                    elif edge_sample['smiles_mark'] == 2:
                        # Edge features are set to 0 or not added
                        edge_features.append(torch.zeros_like(edge_sample['cls_repr']))
                        edge_indices.append([u, v])

            # change to PyTorch Geometric
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.stack(edge_features) if edge_features else None

                # Convert the directed graph to an undirected graph (automatically adding reverse edges)
                edge_index = to_undirected(edge_index)

                # Copy the same edge features for the reversed edges
                if edge_attr is not None:
                    edge_attr = edge_attr.repeat(2, 1)  # 重复特征张量，形状从 [E, F] 变为 [2E, F]
                else:
                    edge_attr = None

                # Collect node features
                node_feat = torch.stack(node_features)

                # Add batch index
                batch_indices = torch.full((node_feat.shape[0],), batch_idx, dtype=torch.long)

                all_nodes.append(node_feat.shape[0])
                all_edges.append(edge_index)
                all_node_features.append(node_feat)
                all_edge_features.append(edge_attr)
                all_batch_indices.extend(batch_indices)

        # Process batch data
        if all_nodes:
            # Merge node features
            x = torch.cat(all_node_features, dim=0)
            # Merge edge index
            edge_index = torch.cat(all_edges, dim=1) if all_edges else None
            # Merge edge features
            edge_attr = torch.cat(all_edge_features, dim=0) if all_edge_features else None
            # batch index
            batch = torch.tensor(all_batch_indices, dtype=torch.long)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        else:
            # No valid nodes, returning empty data
            return Data(x=torch.tensor([], dtype=torch.float32),
                        edge_index=torch.tensor([], dtype=torch.long).t(),
                        batch=torch.tensor([], dtype=torch.long))

    def forward(
            self,
            samples: List[Dict],  # 
            return_repr=False,
            return_atomic_reprs=False,
            **kwargs
    ):
        """
        The forward propagation function processes an input list consisting of 15 dictionaries, 
        constructs a graph structure, and performs regression prediction

        NOTE for AMP (Automatic Mixed Precision):
        To enable mixed precision training for better speed and less memory usage,
        wrap the model call in your training loop with `torch.cuda.amp.autocast()`.
        Example:
        ```python
        from torch.cuda.amp import GradScaler, autocast
        
        scaler = GradScaler()
        # ... in training loop ...
        with autocast():
            output = model(inputs)
            loss = loss_fn(output, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ```
        """
        num_samples = len(samples)
        if num_samples == 0:
            return torch.zeros(0, self.output_dim, device=next(self.parameters()).device)

        real_batch_size = samples[0]['smiles_mark'].numel()
        device = next(self.parameters()).device

        # Micro-batching to balance speed and memory.
        # Instead of processing all 15 samples at once, process them in smaller chunks.
        chunk_size = 1  # Prioritize memory: process one sample at a time. Increase for speed if you have more VRAM.
        
        all_batched_outputs = []

        for i in range(0, num_samples, chunk_size):
            chunk_samples = samples[i:i+chunk_size]
            
            # 1. Find the largest dimension across all atom-related tensors in the current chunk
            max_dim = 0
            variable_keys = ['atom_feat', 'atom_mask', 'degree', 'src_tokens', 'src_coord', 
                             'edge_feat', 'shortest_path', 'pair_type', 'attn_bias']
            for s in chunk_samples:
                for key in variable_keys:
                    if key in s and isinstance(s[key], torch.Tensor):
                        tensor = s[key]
                        if tensor.ndim >= 2:
                            max_dim = max(max_dim, tensor.shape[1])
                        if tensor.ndim >= 3:
                            max_dim = max(max_dim, tensor.shape[2])

            # 2. Pad tensors to max_dim and batch them for the current chunk
            batched_input = {}
            for key in chunk_samples[0].keys():
                if not isinstance(chunk_samples[0].get(key), torch.Tensor):
                    continue

                tensors = [s[key] for s in chunk_samples]

                if key in variable_keys:
                    padded_tensors = []
                    for t in tensors:
                        shape = t.shape
                        padding = None
                        if t.ndim == 2:  # For shapes like (B, N)
                            pad_len = max_dim - shape[1]
                            if pad_len > 0:
                                padding = (0, pad_len)
                        elif t.ndim == 3:  # For shapes like (B, N, D) or (B, N, N)
                            if key in ['atom_feat', 'src_coord']:  # Shape (B, N, D)
                                pad_len = max_dim - shape[1]
                                if pad_len > 0:
                                    padding = (0, 0, 0, pad_len)
                            else:  # Shape (B, N, N)
                                pad_h = max_dim - shape[1]
                                pad_w = max_dim - shape[2]
                                if pad_h > 0 or pad_w > 0:
                                    padding = (0, pad_w, 0, pad_h)
                        elif t.ndim == 4:  # For shapes like (B, N, N, D)
                            pad_h = max_dim - shape[1]
                            pad_w = max_dim - shape[2]
                            if pad_h > 0 or pad_w > 0:
                                padding = (0, 0, 0, pad_w, 0, pad_h)

                        if padding:
                            padded_t = F.pad(t, padding, "constant", self.padding_idx)
                            padded_tensors.append(padded_t)
                        else:
                            padded_tensors.append(t)
                    batched_input[key] = torch.cat(padded_tensors, dim=0)
                else:
                    batched_input[key] = torch.cat(tensors, dim=0)
            
            # 3. Process the chunk through the transformer
            chunk_output = self._process_batch(batched_input)
            all_batched_outputs.append(chunk_output)
        
        # 4. Concatenate results from all chunks
        batched_output = {}
        for key in all_batched_outputs[0].keys():
            if isinstance(all_batched_outputs[0][key], torch.Tensor):
                batched_output[key] = torch.cat([d[key] for d in all_batched_outputs], dim=0)
            else:
                 # Handle non-tensor data if necessary (e.g., smiles_mark)
                batched_output[key] = torch.cat([d[key] for d in all_batched_outputs], dim=0)

        # 5. Reshape and split the results
        processed_samples = [[] for _ in range(real_batch_size)]
        
        reshaped_outputs = {}
        for key, value in batched_output.items():
            if isinstance(value, torch.Tensor):
                # Reshape from (15 * B, ...) to (15, B, ...)
                reshaped_outputs[key] = value.view(num_samples, real_batch_size, *value.shape[1:])
            else:
                reshaped_outputs[key] = value

        for b_idx in range(real_batch_size):
            for s_idx in range(num_samples):
                sample_dict = {key: value[s_idx, b_idx] for key, value in reshaped_outputs.items()}
                processed_samples[b_idx].append(sample_dict)


        # graph_data
        graph_data = self.build_graph_structure(processed_samples)

        # 
        del processed_samples, batched_input, batched_output, reshaped_outputs
        #torch.cuda.empty_cache()

        if graph_data.x.numel() == 0:
            # 
            return torch.zeros(real_batch_size, self.output_dim, device=device)


        # Graph neural network processing
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        batch = graph_data.batch

        # 
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.to(device)
        if isinstance(edge_attr, torch.Tensor):
            edge_attr = edge_attr.to(device)
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)

        # First-layer GAT
        identity1 = x
        x = checkpoint(self.gat1, x, edge_index, edge_attr, use_reentrant=False)
        x = self.norm1(x + identity1)  # 残差+层归一化
        x = F.gelu(x)
        x = F.dropout(x, p=0.05, training=self.training)

        # the second layer GAT
        identity2 = x
        x = checkpoint(self.gat2, x, edge_index, edge_attr, use_reentrant=False)
        x = self.norm2(x + identity2)  # 残差+层归一化
        x = F.gelu(x)
        x = F.dropout(x, p=0.05, training=self.training)

        # pooling
        mean_feat = global_mean_pool(x, batch)  # 全局平均池化
        max_feat = global_max_pool(x, batch)  # 全局最大池化
        sum_feat = global_add_pool(x, batch)  # 全局求和池化
        graph_repr = (mean_feat + max_feat + sum_feat) / 3  # 三种结果取平均

        # regression prediction
        logits = self.graph_mlp(graph_repr)

        return logits

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

    def batch_collate_fn(self, samples):
        """
        Custom collate function for batch processing non-MOF data.

        :param samples: A list of sample data.

        :return: A tuple containing a batch dictionary and labels.
        """
        batch = {}
        for k in samples[0][0].keys():
            if k == 'atom_feat':
                v = pad_coords(
                    [torch.tensor(s[0][k]) for s in samples],
                    pad_idx=self.padding_idx,
                    dim=8,
                )
            elif k == 'atom_mask':
                v = pad_1d_tokens(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'edge_feat':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples],
                    pad_idx=self.padding_idx,
                    dim=3,
                )
            elif k == 'shortest_path':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'degree':
                v = pad_1d_tokens(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'pair_type':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples],
                    pad_idx=self.padding_idx,
                    dim=2,
                )
            elif k == 'attn_bias':
                v = pad_2d(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'src_tokens':
                v = pad_1d_tokens(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            elif k == 'src_coord':
                v = pad_coords(
                    [torch.tensor(s[0][k]) for s in samples], pad_idx=self.padding_idx
                )
            batch[k] = v
        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        return batch, label

    def batch_collate_fn2(self, samples):
        """
        Custom collate function for batch processing non-MOF data.

        :param samples: A list of sample data.

        :return: A tuple containing a batch dictionary and labels.
        """
        batch = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        for idx in range(15):
            for k in samples[0][0][0].keys():
                if k == 'atom_feat':
                    v = pad_coords(
                        [torch.tensor(s[0][idx][k]) for s in samples],
                        pad_idx=self.padding_idx,
                        dim=8,
                    )
                elif k == 'atom_mask':
                    v = pad_1d_tokens(
                        [torch.tensor(s[0][idx][k]) for s in samples], pad_idx=self.padding_idx
                    )
                elif k == 'edge_feat':
                    v = pad_2d(
                        [torch.tensor(s[0][idx][k]) for s in samples],
                        pad_idx=self.padding_idx,
                        dim=3,
                    )
                elif k == 'shortest_path':
                    v = pad_2d(
                        [torch.tensor(s[0][idx][k]) for s in samples], pad_idx=self.padding_idx
                    )
                elif k == 'degree':
                    v = pad_1d_tokens(
                        [torch.tensor(s[0][idx][k]) for s in samples], pad_idx=self.padding_idx
                    )
                elif k == 'pair_type':
                    v = pad_2d(
                        [torch.tensor(s[0][idx][k]) for s in samples],
                        pad_idx=self.padding_idx,
                        dim=2,
                    )
                elif k == 'attn_bias':
                    v = pad_2d(
                        [torch.tensor(s[0][idx][k]) for s in samples], pad_idx=self.padding_idx
                    )
                elif k == 'src_tokens':
                    v = pad_1d_tokens(
                        [torch.tensor(s[0][idx][k]) for s in samples], pad_idx=self.padding_idx
                    )
                elif k == 'src_coord':
                    v = pad_coords(
                        [torch.tensor(s[0][idx][k]) for s in samples], pad_idx=self.padding_idx
                    )
                batch[idx][k] = v
                #print(f"batch设备: {batch[idx][k].device}")  cpu
            batch[idx]['artificial_feat'] = torch.tensor([s[1][idx] for s in samples]).float()
            batch[idx]['smiles_mark'] = torch.tensor([s[2][idx] for s in samples]).long()
        #try:
            label = torch.tensor([s[-1] for s in samples])
        #except:
            #label = None
            #print(f"label设备: {label.device}")  cpu
        return batch, label


class LinearHead(nn.Module):
    """Linear head."""

    def __init__(
        self,
        input_dim,
        num_classes,
        pooler_dropout,
    ):
        """
        Initialize the Linear head.

        :param input_dim: Dimension of input features.
        :param num_classes: Number of classes for output.
        """
        super().__init__()
        self.out_proj = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(p=pooler_dropout)

    def forward(self, features, **kwargs):
        """
        Forward pass for the Linear head.

        :param features: Input features.

        :return: Output from the Linear head.
        """
        x = features
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """
    A neural network module used for simple classification tasks. It consists of a two-layered linear network
    with a nonlinear activation function in between.

    Attributes:
        - linear1: The first linear layer.
        - linear2: The second linear layer that outputs to the desired dimensions.
        - activation_fn: The nonlinear activation function.
    """

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        """
        Initializes the NonLinearHead module.

        :param input_dim: Dimension of the input features.
        :param out_dim: Dimension of the output.
        :param activation_fn: The activation function to use.
        :param hidden: Dimension of the hidden layer; defaults to the same as input_dim if not provided.
        """
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the module.

        :return: Tensor after passing through the network.
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


@torch.jit.script
def gaussian(x, mean, std):
    """
    Gaussian function implemented for PyTorch tensors.

    :param x: The input tensor.
    :param mean: The mean for the Gaussian function.
    :param std: The standard deviation for the Gaussian function.

    :return: The output tensor after applying the Gaussian function.
    """
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


def get_activation_fn(activation):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class GaussianLayer(nn.Module):
    """
    A neural network module implementing a Gaussian layer, useful in graph neural networks.

    Attributes:
        - K: Number of Gaussian kernels.
        - means, stds: Embeddings for the means and standard deviations of the Gaussian kernels.
        - mul, bias: Embeddings for scaling and bias parameters.
    """

    def __init__(self, K=128, edge_types=1024):
        """
        Initializes the GaussianLayer module.

        :param K: Number of Gaussian kernels.
        :param edge_types: Number of different edge types to consider.

        :return: An instance of the configured Gaussian kernel and edge types.
        """
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        """
        Forward pass of the GaussianLayer.

        :param x: Input tensor representing distances or other features.
        :param edge_type: Tensor indicating types of edges in the graph.

        :return: Tensor transformed by the Gaussian layer.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NumericalEmbed(nn.Module):
    """
    Numerical embedding module, typically used for embedding edge features in graph neural networks.

    Attributes:
        - K: Output dimension for embeddings.
        - mul, bias, w_edge: Embeddings for transformation parameters.
        - proj: Projection layer to transform inputs.
        - ln: Layer normalization.
    """

    def __init__(self, K=128, edge_types=1024, activation_fn='gelu'):
        """
        Initializes the NonLinearHead.

        :param input_dim: The input dimension of the first layer.
        :param out_dim: The output dimension of the second layer.
        :param activation_fn: The activation function to use.
        :param hidden: The dimension of the hidden layer; defaults to input_dim if not specified.
        """
        super().__init__()
        self.K = K
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.w_edge = nn.Embedding(edge_types, K)

        self.proj = NonLinearHead(1, K, activation_fn, hidden=2 * K)
        self.ln = nn.LayerNorm(K)

        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        nn.init.kaiming_normal_(self.w_edge.weight)

    def forward(self, x, edge_type):  # edge_type, atoms
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the classification head.

        :return: The output tensor after passing through the layers.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        w_edge = self.w_edge(edge_type).type_as(x)
        edge_emb = w_edge * torch.sigmoid(mul * x.unsqueeze(-1) + bias)

        edge_proj = x.unsqueeze(-1).type_as(self.mul.weight)
        edge_proj = self.proj(edge_proj)
        edge_proj = self.ln(edge_proj)

        h = edge_proj + edge_emb
        h = h.type_as(self.mul.weight)
        return h


def molecule_architecture(model_size='84m'):
    args = Dict()
    if model_size == '84m':
        args.num_encoder_layers = 12
        args.encoder_embed_dim = 768
        args.num_attention_heads = 48
        args.ffn_embedding_dim = 768
        args.encoder_attention_heads = 48
    elif model_size == '164m':
        args.num_encoder_layers = 24
        args.encoder_embed_dim = 768
        args.num_attention_heads = 48
        args.ffn_embedding_dim = 768
        args.encoder_attention_heads = 48
    elif model_size == '310m':
        args.num_encoder_layers = 32
        args.encoder_embed_dim = 1024
        args.num_attention_heads = 64
        args.ffn_embedding_dim = 1024
        args.encoder_attention_heads = 64
    elif model_size == '570m':
        args.num_encoder_layers = 32
        args.encoder_embed_dim = 1536
        args.num_attention_heads = 96
        args.ffn_embedding_dim = 1536
        args.encoder_attention_heads = 96
    elif model_size == '1.1B':
        args.num_encoder_layers = 64
        args.encoder_embed_dim = 1536
        args.num_attention_heads = 96
        args.ffn_embedding_dim = 1536
        args.encoder_attention_heads = 96
    else:
        raise ValueError('Current not support data type: {}'.format(model_size))
    args.pair_embed_dim = 512
    args.pair_hidden_dim = 64
    args.dropout = 0.1
    args.attention_dropout = 0.1
    args.activation_dropout = 0.0
    args.activation_fn = "gelu"
    args.droppath_prob = 0.0
    args.pair_dropout = 0.25
    args.backbone = "transformer"
    args.gaussian_std_width = 1.0
    args.gaussian_mean_start = 0.0
    args.gaussian_mean_stop = 9.0
    args.pooler_dropout = 0.0
    args.pooler_activation_fn = "tanh"
    return args
