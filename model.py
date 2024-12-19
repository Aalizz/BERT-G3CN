from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import scipy.sparse as sp

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.configuration_utils import PretrainedConfig

from transformers.activations import get_activation
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from configuration_vgcn_bert import VGCNBertConfig

logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "zhibinlu/vgcn-bert-distilbert-base-uncased"
_CONFIG_FOR_DOC = "VGCNBertConfig"

VGCNBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "zhibinlu/vgcn-bert-distilbert-base-uncased",
    # See all VGCN-BERT models at https://huggingface.co/models?filter=vgcn-bert
]

# Word Graph construction utils #

ENGLISH_STOP_WORDS = frozenset(
    {
        "herself","each","him","been","only","yourselves","into","where","them","very","we","that","re","too","some","what","those","me","whom","have",
        "yours",
        "an",
        "during",
        "any",
        "nor",
        "ourselves",
        "has",
        "do",
        "when",
        "about",
        "same",
        "our",
        "then",
        "himself",
        "their",
        "all",
        "no",
        "a",
        "hers",
        "off",
        "why",
        "how",
        "more",
        "between",
        "until",
        "not",
        "over",
        "your",
        "by",
        "here",
        "most",
        "above",
        "up",
        "of",
        "is",
        "after",
        "from",
        "being",
        "i",
        "as",
        "other",
        "so",
        "her",
        "ours",
        "on",
        "because",
        "against",
        "and",
        "out",
        "had",
        "these",
        "at",
        "both",
        "down",
        "you",
        "can",
        "she",
        "few",
        "the",
        "if",
        "it",
        "to","but","its","be","he","once","further","such","there","through","are","themselves","which","in","now","his","yourself","this","were","below","should","my","myself","am","or","while","itself","again","with","they","will","own","than","before","under","was","for","who",
    }
)

def _normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) # 归一化

def _scipy_to_torch(sparse):
    sparse = sparse.tocoo() if sparse.getformat() != "coo" else sparse
    i = torch.LongTensor(np.vstack((sparse.row, sparse.col)))
    v = torch.from_numpy(sparse.data)
    return torch.sparse_coo_tensor(i, v, torch.Size(sparse.shape)).coalesce()

def _delete_special_terms(words: list, terms: set):
    return set([w for w in words if w not in terms])

def _build_pmi_graph(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    window_size=20,
    algorithm="npmi",
    edge_threshold=0.0,
    remove_stopwords=False,
    min_freq_to_keep=2,
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Build statistical word graph from text samples using PMI or NPMI algorithm.
    """
    words_to_remove = (
        set({"[CLS]", "[SEP]"}).union(ENGLISH_STOP_WORDS) if remove_stopwords else set({"[CLS]", "[SEP]"})
    )
    vocab_counter = Counter()
    texts_words = []
    for t in texts:
        words = tokenizer.tokenize(t)
        words = _delete_special_terms(words, words_to_remove)
        if len(words) > 0:
            vocab_counter.update(Counter(words))
            texts_words.append(words)

    new_vocab_counter = Counter({"[PAD]": 0})
    new_vocab_counter.update(
        Counter({k: v for k, v in vocab_counter.items() if v >= min_freq_to_keep})
        if min_freq_to_keep > 1
        else vocab_counter
    )
    vocab_counter = new_vocab_counter

    # Generate new texts by removing words with freq<n
    if min_freq_to_keep > 1:
        texts_words = [list(filter(lambda w: vocab_counter[w] >= min_freq_to_keep, words)) for words in texts_words]
    texts = [" ".join(words).strip() for words in texts_words if len(words) > 0]

    vocab_size = len(vocab_counter)
    vocab = list(vocab_counter.keys())
    assert vocab[0] == "[PAD]"
    vocab_indices = {k: i for i, k in enumerate(vocab)}

    windows = []
    for t in texts:
        words = t.split()
        word_ids = [vocab_indices[w] for w in words]
        length = len(word_ids)
        if length <= window_size:
            windows.append(word_ids)
        else:
            for j in range(length - window_size + 1):
                word_ids = word_ids[j : j + window_size]
                windows.append(word_ids)

    vocab_window_counter = Counter()
    word_pair_window_counter = Counter()
    for word_ids in windows:
        word_ids = list(set(word_ids))
        vocab_window_counter.update(Counter(word_ids))
        word_pair_window_counter.update(
            Counter(
                [
                    f(i, j)
                    # (word_ids[i], word_ids[j])
                    for i in range(1, len(word_ids))
                    for j in range(i)
                    # adding inverse pair
                    for f in (lambda x, y: (word_ids[x], word_ids[y]), lambda x, y: (word_ids[y], word_ids[x]))
                ]
            )
        )

    # Calculate NPMI or PMI
    vocab_adj_row = []
    vocab_adj_col = []
    vocab_adj_weight = []

    total_windows = len(windows)
    for wid_pair in word_pair_window_counter.keys():
        i, j = wid_pair
        pair_count = word_pair_window_counter[wid_pair]
        i_count = vocab_window_counter[i]
        j_count = vocab_window_counter[j]
        if algorithm == "npmi":
            # NPMI calculation
            value = (math.log((i_count * j_count) / (total_windows ** 2)) / math.log(pair_count / total_windows) - 1)
        else:  # pmi
            value = math.log((pair_count / total_windows) / (i_count * j_count / (total_windows ** 2)))
        if value > edge_threshold:
            vocab_adj_row.append(i)
            vocab_adj_col.append(j)
            vocab_adj_weight.append(value)

    vocab_adj = sp.csr_matrix(
        (vocab_adj_weight, (vocab_adj_row, vocab_adj_col)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    )
    vocab_adj.setdiag(1.0)

    assert vocab_adj[0, :].sum() == 1
    assert vocab_adj[:, 0].sum() == 1
    vocab_adj[:, 0] = 0
    vocab_adj[0, :] = 0

    wgraph_id_to_tokenizer_id_map = {v: tokenizer.vocab[k] for k, v in vocab_indices.items()}
    wgraph_id_to_tokenizer_id_map = dict(sorted(wgraph_id_to_tokenizer_id_map.items()))

    return (
        vocab_adj,
        vocab_indices,
        wgraph_id_to_tokenizer_id_map,
    )

def _build_predefined_graph(
    words_relations: List[Tuple[str, str, float]], tokenizer: PreTrainedTokenizerBase, remove_stopwords: bool = False
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Build pre-defined wgraph from a list of word pairs and their pre-defined relations (edge value).
    """

    words_to_remove = (
        set({"[CLS]", "[SEP]"}).union(ENGLISH_STOP_WORDS) if remove_stopwords else set({"[CLS]", "[SEP]"})
    )
    vocab_counter = Counter({"[PAD]": 0})
    word_pairs = {}
    for w1, w2, v in words_relations:
        w1_subwords = tokenizer.tokenize(w1)
        w1_subwords = _delete_special_terms(w1_subwords, words_to_remove)
        w2_subwords = tokenizer.tokenize(w2)
        w2_subwords = _delete_special_terms(w2_subwords, words_to_remove)
        vocab_counter.update(Counter(w1_subwords))
        vocab_counter.update(Counter(w2_subwords))
        for sw1 in w1_subwords:
            for sw2 in w2_subwords:
                if sw1 != sw2:
                    word_pairs.setdefault((sw1, sw2), v)

    vocab_size = len(vocab_counter)
    vocab = list(vocab_counter.keys())
    assert vocab[0] == "[PAD]"
    vocab_indices = {k: i for i, k in enumerate(vocab)}

    vocab_adj_row = []
    vocab_adj_col = []
    vocab_adj_weight = []
    for (w1, w2), v in word_pairs.items():
        vocab_adj_row.append(vocab_indices[w1])
        vocab_adj_col.append(vocab_indices[w2])
        vocab_adj_weight.append(v)
        # adding inverse
        vocab_adj_row.append(vocab_indices[w2])
        vocab_adj_col.append(vocab_indices[w1])
        vocab_adj_weight.append(v)

    # Build vocabulary adjacency matrix
    vocab_adj = sp.csr_matrix(
        (vocab_adj_weight, (vocab_adj_row, vocab_adj_col)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    )
    vocab_adj.setdiag(1.0)

    assert vocab_adj[0, :].sum() == 1
    assert vocab_adj[:, 0].sum() == 1
    vocab_adj[:, 0] = 0
    vocab_adj[0, :] = 0

    wgraph_id_to_tokenizer_id_map = {v: tokenizer.vocab[k] for k, v in vocab_indices.items()}
    wgraph_id_to_tokenizer_id_map = dict(sorted(wgraph_id_to_tokenizer_id_map.items()))

    return (
        vocab_adj,
        vocab_indices,
        wgraph_id_to_tokenizer_id_map,
    )

# TODO: build knowledge graph from a list of RDF triples

class WordGraphBuilder:
    """
    Word graph based on adjacency matrix, construct from text samples or pre-defined word-pair relations
    You may (or not) first preprocess the text before build the graph,
    e.g. Stopword removal, String cleaning, Stemming, Nomolization, Lemmatization
    Params:
        `rows`: List[str] of text samples, or pre-defined word-pair relations: List[Tuple[str, str, float]]
        `tokenizer`: The same pretrained tokenizer that is used for the model late.
        `window_size`:  Available only for statistics generation (rows is text samples).
            Size of the sliding window for collecting the pieces of text
            and further calculate the NPMI value, default is 20.
        `algorithm`:  Available only for statistics generation (rows is text samples) -- "npmi" or "pmi", default is "npmi".
        `edge_threshold`: Available only for statistics generation (rows is text samples). Graph edge value threshold, default is 0.0. Edge value is between -1 to 1.
        `remove_stopwords`: Build word graph with the words that are not stopwords, default is False.
        `min_freq_to_keep`: Available only for statistics generation (rows is text samples). Build word graph with the words that occurred at least n times in the corpus, default is 2.

    Properties:
        `adjacency_matrix`: scipy.sparse.csr_matrix, the word graph in sparse adjacency matrix form.
        `vocab_indices`: indices of word graph vocabulary words.
        `wgraph_id_to_tokenizer_id_map`: map from word graph vocabulary word id to tokenizer vocabulary word id.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        rows: list,
        tokenizer: PreTrainedTokenizerBase,
        window_size=20,
        algorithm="npmi",
        edge_threshold=0.0,
        remove_stopwords=False,
        min_freq_to_keep=2,
    ):
        if isinstance(rows[0], tuple):
            (
                adjacency_matrix,
                _,
                wgraph_id_to_tokenizer_id_map,
            ) = _build_predefined_graph(rows, tokenizer, remove_stopwords)
        else:
            (
                adjacency_matrix,
                _,
                wgraph_id_to_tokenizer_id_map,
            ) = _build_pmi_graph(
                rows, tokenizer, window_size, algorithm, edge_threshold, remove_stopwords, min_freq_to_keep
            )

        adjacency_matrix = _scipy_to_torch(_normalize_adj(adjacency_matrix)) if adjacency_matrix is not None else None
        return adjacency_matrix, wgraph_id_to_tokenizer_id_map

class VgcnParameterList(nn.ParameterList):
    def __init__(self, values=None, requires_grad=True) -> None:
        super().__init__(values)
        self.requires_grad = requires_grad

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        keys = filter(lambda x: x.startswith(prefix), state_dict.keys())
        for k in keys:
            self.append(nn.Parameter(state_dict[k], requires_grad=self.requires_grad))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        for i in range(len(self)):
            if self[i].layout is torch.sparse_coo and not self[i].is_coalesced():
                self[i] = self[i].coalesce()
            self[i].requires_grad = self.requires_grad

# ----------------------------------------------- #
#             VGCN-BERT MODELING                  #
# ----------------------------------------------- #

class VocabGraphConvolution(nn.Module):
    """Vocabulary GCN module supporting multiple graph types."""

    def __init__(
            self,
            hid_dim: int,
            out_dim: int,
            activation=None,
            dropout_rate=0.1,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.fc_hg = nn.Linear(hid_dim, out_dim)
        self.fc_hg._is_vgcn_linear = True
        self.activation = get_activation(activation) if activation else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(
            self,
            word_embeddings: nn.Embedding,
            input_ids: torch.Tensor,
            adj: torch.sparse_coo_tensor,
            map_: Dict[int, int]
    ):
        """
        参数:
            word_embeddings (nn.Embedding): 词嵌入层。
            input_ids (torch.Tensor): 输入的 token IDs，形状为 (batch_size, seq_length)。
            adj (torch.sparse_coo_tensor): 图的邻接矩阵。
            map_ (Dict[int, int]): 图 ID 到 tokenizer ID 的映射。
        返回:
            torch.Tensor: 图嵌入，形状为 (batch_size, out_dim)。
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        graph_embeds = torch.zeros(batch_size, self.out_dim, device=device)

        for i in range(batch_size):
            if adj[i] is not None:
                current_adj = adj[i].to(device)  # 假设 adj 是 batch_size 个图的列表或堆叠
                current_map = map_[i]  # 假设 map_ 是 batch_size 个图的列表

                if len(current_map) == 0:
                    # 处理空图
                    graph_embeds[i] = torch.zeros(self.out_dim, device=device)
                    continue

                # 获取图的 tokenizer ID 列表
                tokenizer_ids = list(current_map.values())
                gv_ids = torch.tensor(tokenizer_ids, dtype=torch.long, device=device)  # (num_graph_vocab,)

                # 获取词嵌入
                gvocab_ev = word_embeddings(gv_ids)  # (num_graph_vocab, emb_dim)

                # 将稀疏邻接矩阵转换为稠密矩阵
                adj_dense = current_adj.to_dense()  # (num_nodes, num_nodes)

                # 图卷积操作：adj * embeddings
                H_vh = torch.matmul(adj_dense, gvocab_ev)  # (num_nodes, emb_dim)

                # 聚合图嵌入（例如，取平均）
                graph_embed = H_vh.mean(dim=0)  # (emb_dim)
                graph_embed = self.fc_hg(graph_embed)  # (out_dim)

                # 激活和 dropout
                if self.activation:
                    graph_embed = self.activation(graph_embed)
                if self.dropout:
                    graph_embed = self.dropout(graph_embed)

                graph_embeds[i] = graph_embed
            else:
                continue

        return graph_embeds

def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)

def _create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()

class VGCNEmbeddings(nn.Module):
    """构建词嵌入、多个 VGCN 图嵌入、位置嵌入和 token_type 嵌入的组合。"""

    def __init__(
            self,
            config: PretrainedConfig,
    ):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)

        # 初始化不同类型的图卷积模块
        self.vgcn_cooccurrence = VocabGraphConvolution(
            hid_dim=config.vgcn_hidden_dim,
            out_dim=config.vgcn_graph_embds_dim,
            activation=config.vgcn_activation,
            dropout_rate=config.vgcn_dropout,
        )
        self.vgcn_hetero = VocabGraphConvolution(
            hid_dim=config.vgcn_hidden_dim,
            out_dim=config.vgcn_hetero_graph_embds_dim,
            activation=config.vgcn_activation,
            dropout_rate=config.vgcn_dropout,
        )
        self.vgcn_isg = VocabGraphConvolution(
            hid_dim=config.vgcn_hidden_dim,
            out_dim=config.vgcn_isg_graph_embds_dim,
            activation=config.vgcn_activation,
            dropout_rate=config.vgcn_dropout,
        )

        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )


        total_graph_dim = config.vgcn_graph_embds_dim + config.vgcn_hetero_graph_embds_dim + config.vgcn_isg_graph_embds_dim

        self.projection = nn.Linear(config.dim + total_graph_dim, config.dim)
        # 初始化投影层的权重
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)


        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            input_ids: torch.Tensor,
            cooccurrence_adjs: List[torch.sparse_coo_tensor],
            cooccurrence_maps: List[Dict[int, int]],
            hetero_adj: torch.sparse_coo_tensor,
            hetero_map: Dict[int, int],
            isg_adj: torch.sparse_coo_tensor,
            isg_map: Dict[int, int],
            input_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数：
            input_ids (torch.Tensor): (batch_size, seq_length)
            cooccurrence_adjs (List[torch.sparse_coo_tensor]): 每个样本的 Cooccurrence 图的邻接矩阵列表。
            cooccurrence_maps (List[Dict[int, int]]): 每个样本的 Cooccurrence 图的映射列表。
            hetero_adj (torch.sparse_coo_tensor): 异构图的邻接矩阵（共享全局异构图）。
            hetero_map (Dict[int, int]): 异构图的映射（共享全局异构图）。
            isg_adj (torch.sparse_coo_tensor): ISG图的邻接矩阵（共享全局ISG图）。
            isg_map (Dict[int, int]): ISG图的映射（共享全局ISG图）。
            input_embeds (Optional[torch.Tensor]): 预计算的嵌入。

        返回：
            torch.Tensor: 投影后的嵌入，形状为 (batch_size, seq_length, config.dim)
        """
        if input_embeds is None:
            input_embeds = self.word_embeddings(input_ids)  # (batch_size, seq_length, dim)

        seq_length = input_embeds.size(1)

        # 位置嵌入
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (batch_size, seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (batch_size, seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (batch_size, seq_length, dim)

        # 图嵌入
        graph_embeds_cooccurrence = self.vgcn_cooccurrence(
            self.word_embeddings,
            input_ids,
            cooccurrence_adjs,
            cooccurrence_maps
        )

        graph_embeds_hetero = self.vgcn_hetero(
            self.word_embeddings,
            input_ids,
            hetero_adj,
            hetero_map
        )

        graph_embeds_isg = self.vgcn_isg(
            self.word_embeddings,
            input_ids,
            isg_adj,
            isg_map
        )


        graph_embeds = torch.cat([graph_embeds_cooccurrence, graph_embeds_hetero, graph_embeds_isg], dim=1)


        graph_embeds = graph_embeds.unsqueeze(1).expand(-1, seq_length, -1)

        embeddings = torch.cat([embeddings, graph_embeds], dim=2)

        embeddings = self.projection(embeddings)

        # 应用 LayerNorm 和 Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        if self.dim % self.n_heads != 0:
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads: List[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights
            context: torch.tensor(bs, seq_length, dim) Contextualized layer.
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)

class FFN(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.activation = get_activation(config.activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor:
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights
            ffn_output: torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:
            if type(sa_output) != tuple:
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

class Transformer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer
            all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i] if head_mask is not None else None, output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

class VGCNBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VGCNBertConfig
    load_tf_weights = None
    base_model_prefix = "vgcn_bert"

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if getattr(module, "_is_vgcn_linear", False):
                if self.config.vgcn_weight_init_mode == "transparent":
                    module.weight.data.fill_(1.0)
                elif self.config.vgcn_weight_init_mode == "normal":
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                elif self.config.vgcn_weight_init_mode == "uniform":
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                else:
                    raise ValueError(f"Unknown VGCN-BERT weight init mode: {self.config.vgcn_weight_init_mode}.")
                if module.bias is not None:
                    module.bias.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VgcnParameterList):
            if getattr(module, "_is_vgcn_weights", False):
                for p in module:
                    if self.config.vgcn_weight_init_mode == "transparent":
                        nn.init.constant_(p, 1.0)
                    elif self.config.vgcn_weight_init_mode == "normal":
                        nn.init.normal_(p, mean=0.0, std=self.config.initializer_range)
                    elif self.config.vgcn_weight_init_mode == "uniform":
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    else:
                        raise ValueError(f"Unknown VGCN-BERT weight init mode: {self.config.vgcn_weight_init_mode}.")

VGCNBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VGCNBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VGCNBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare VGCN-BERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    VGCNBERT_START_DOCSTRING,
)
class VGCNBertModel(VGCNBertPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__(config)

        self.embeddings = VGCNEmbeddings(config)
        self.transformer = Transformer(config)  # Encoder

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cooccurrence_adjs: Optional[List[torch.sparse_coo_tensor]] = None,
        cooccurrence_maps: Optional[List[Dict[int, int]]] = None,
        hetero_adj: Optional[torch.sparse_coo_tensor] = None,
        hetero_map: Optional[Dict[int, int]] = None,
        isg_adj: Optional[torch.sparse_coo_tensor] = None,
        isg_map: Optional[Dict[int, int]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        参数：
            input_ids (torch.Tensor): (batch_size, seq_length)
            attention_mask (torch.Tensor): (batch_size, seq_length)
            cooccurrence_adjs (List[torch.sparse_coo_tensor], optional): 每个样本的 Cooccurrence 图的邻接矩阵列表。
            cooccurrence_maps (List[Dict[int, int]], optional): 每个样本的 Cooccurrence 图的映射列表。
            hetero_adj (torch.sparse_coo_tensor, optional): 异构图的邻接矩阵（共享全局异构图）。
            hetero_map (Dict[int, int], optional): 异构图的映射（共享全局异构图）。
            isg_adj (torch.sparse_coo_tensor, optional): ISG图的邻接矩阵（共享全局ISG图）。
            isg_map (Dict[int, int], optional): ISG图的映射（共享全局ISG图）。
            head_mask (torch.Tensor, optional): Head mask。
            inputs_embeds (torch.Tensor, optional): 输入嵌入。
            output_attentions (bool, optional): 是否输出注意力权重。
            output_hidden_states (bool, optional): 是否输出隐藏状态。
            return_dict (bool, optional): 是否返回字典。

        返回：
            BaseModelOutput: Transformer 的输出。
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时指定 input_ids 和 inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("必须指定 input_ids 或 inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (batch_size, seq_length)

        # 准备 head mask（如果需要）
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = self.embeddings(
            input_ids=input_ids,
            cooccurrence_adjs=cooccurrence_adjs,
            cooccurrence_maps=cooccurrence_maps,
            hetero_adj=hetero_adj,
            hetero_map=hetero_map,
            isg_adj=isg_adj,
            isg_map=isg_map,
            input_embeds=inputs_embeds
        )  # (batch_size, seq_length, dim)

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

@dataclass
class VGCNBertForSequenceClassificationOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@add_start_docstrings(
    """VGCN-Bert Model with a `sequence classification` head on top (a linear layer on top of the pooled output).""",
    VGCNBERT_START_DOCSTRING,
)
class VGCNBertForSequenceClassification(VGCNBertPreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.vgcn_bert = VGCNBertModel(config)
        # 计算总图嵌入维度
        total_graph_dim = config.vgcn_graph_embds_dim + config.vgcn_hetero_graph_embds_dim + config.vgcn_isg_graph_embds_dim
        self.pre_classifier = nn.Linear(config.dim, config.dim)  # 由于已经投影回 config.dim，无需额外维度
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cooccurrence_adjs: Optional[List[torch.sparse_coo_tensor]] = None,
        cooccurrence_maps: Optional[List[Dict[int, int]]] = None,
        hetero_adj: Optional[torch.sparse_coo_tensor] = None,  # 修改为单个全局异构图
        hetero_map: Optional[Dict[int, int]] = None,         # 修改为单个全局异构图映射
        isg_adj: Optional[torch.sparse_coo_tensor] = None,    # 修改为单个全局ISG图
        isg_map: Optional[Dict[int, int]] = None,           # 修改为单个全局ISG图映射
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        """
        参数：
            cooccurrence_adjs (List[torch.sparse_coo_tensor], optional):
                每个样本的 Cooccurrence 图的邻接矩阵列表。
            cooccurrence_maps (List[Dict[int, int]], optional):
                每个样本的 Cooccurrence 图的映射列表。
            hetero_adj (torch.sparse_coo_tensor, optional): 异构图的邻接矩阵（共享全局异构图）。
            hetero_map (Dict[int, int], optional): 异构图的映射（共享全局异构图）。
            isg_adj (torch.sparse_coo_tensor, optional): ISG图的邻接矩阵（共享全局ISG图）。
            isg_map (Dict[int, int], optional): ISG图的映射（共享全局ISG图）。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vgcn_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cooccurrence_adjs=cooccurrence_adjs,
            cooccurrence_maps=cooccurrence_maps,
            hetero_adj=hetero_adj,
            hetero_map=hetero_map,
            isg_adj=isg_adj,
            isg_map=isg_map,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = outputs.last_hidden_state 
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
