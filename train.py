import gc
import numpy as np
import pandas as pd
import networkx as nx
import torch
import scipy.sparse as sp
import nltk
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from model import (
    _scipy_to_torch,
    _normalize_adj,
    VGCNBertConfig,
    VGCNBertForSequenceClassification,
    WordGraphBuilder
)
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from functools import partial
from torch.cuda.amp import autocast, GradScaler
from text2graphapi.src.Heterogeneous import Heterogeneous
from text2graphapi.src.IntegratedSyntacticGraph import ISG

# 确保下载所需的 nltk 资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def graph_to_scipy_sparse_matrix(graph, tokenizer):
    if isinstance(graph, dict):
        graph = graph.get('graph', graph)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected NetworkX graph, but got {type(graph)}")

    nodes = list(graph.nodes())

    if len(nodes) == 0:
        print("Empty graph encountered, creating default adjacency and map")
        adj = sp.csr_matrix((1, 1), dtype=np.float32)
        wgraph_id_to_tokenizer_id_map = {0: tokenizer.vocab.get('[UNK]', 0)}
        return adj, wgraph_id_to_tokenizer_id_map

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    row, col, data = [], [], []
    for edge in graph.edges(data=True):
        src = node_to_idx[edge[0]]
        dst = node_to_idx[edge[1]]
        weight = edge[2].get('weight', 1.0)
        row.append(src)
        col.append(dst)
        data.append(weight)

    size = len(nodes)
    adj = sp.coo_matrix((data, (row, col)), shape=(size, size), dtype=np.float32)

    adj = _normalize_adj(adj.tocsr())

    vocab = [idx_to_node[idx] for idx in range(size)]
    wgraph_id_to_tokenizer_id_map = {}
    for graph_id, word in enumerate(vocab):
        wgraph_id_to_tokenizer_id_map[graph_id] = tokenizer.vocab.get(word, tokenizer.vocab.get('[UNK]', 0))

    return adj, dict(sorted(wgraph_id_to_tokenizer_id_map.items()))

class SpamDataset(Dataset):
    def __init__(
            self,
            messages,
            labels,
            tokenizer,
            graph_builder,
            hetero_adj,
            hetero_map,
            isg_adj,
            isg_map,
            max_length=128
    ):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.graph_builder = graph_builder
        self.hetero_adj = hetero_adj
        self.hetero_map = hetero_map
        self.isg_adj = isg_adj
        self.isg_map = isg_map
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        text = self.messages[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        cooccurrence_adj, cooccurrence_map = self.graph_builder(rows=[text], tokenizer=self.tokenizer)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'cooccurrence_adj': cooccurrence_adj,
            'cooccurrence_map': cooccurrence_map,
            'hetero_adj': self.hetero_adj,
            'hetero_map': self.hetero_map,
            'isg_adj': self.isg_adj,
            'isg_map': self.isg_map
        }

# 定义自定义的 collate_fn
def custom_collate_fn(batch):
    batch_size = len(batch)

    batch_input_ids = torch.stack([item['input_ids'] for item in batch])
    batch_attention_mask = torch.stack([item['attention_mask'] for item in batch])
    batch_labels = torch.stack([item['labels'] for item in batch])

    # 图嵌入保存在列表中
    batch_cooccurrence_adjs = [item['cooccurrence_adj'] for item in batch]
    batch_cooccurrence_maps = [item['cooccurrence_map'] for item in batch]

    # 异构图和ISG图为全局共享，直接取第一个并重复
    batch_hetero_adj = [batch[0]['hetero_adj']] * batch_size
    batch_hetero_map = [batch[0]['hetero_map']] * batch_size
    batch_isg_adj = [batch[0]['isg_adj']] * batch_size
    batch_isg_map = [batch[0]['isg_map']] * batch_size

    return {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask,
        'labels': batch_labels,
        'cooccurrence_adjs': batch_cooccurrence_adjs,
        'cooccurrence_maps': batch_cooccurrence_maps,
        'hetero_adj': batch_hetero_adj,
        'hetero_map': batch_hetero_map,
        'isg_adj': batch_isg_adj,
        'isg_map': batch_isg_map
    }

# 加载数据和初始化图构建器
data = pd.read_csv('../processed_spam2.csv')

# 转换为 DataFrame
df = pd.DataFrame(data)

# 显示结果
messages = df['Message'].tolist()
labels = df['Label'].tolist()

# 初始化图构建器
word_graph_builder = WordGraphBuilder()

# 初始化各类图构建器
to_hetero_graph = Heterogeneous(
    graph_type='Graph',
    window_size=20,
    apply_prep=True,
    language='en',
    output_format='networkx'
)

to_isg_graph = ISG(
    graph_type='DiGraph',
    language='en',
    apply_prep=True,
    output_format='networkx'
)

# 定义两个不同的输入数据
texts_for_cooccurrence = [msg for msg in messages]
corpus_docs = [{'id': idx, 'doc': msg} for idx, msg in enumerate(messages)]

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')

# 使用 VGCN-BERT 方法构建共现图
cooccurrence_adj, cooccurrence_map = word_graph_builder(
    rows=texts_for_cooccurrence,
    tokenizer=tokenizer,
    window_size=20,
    algorithm="npmi",
    edge_threshold=0.0,
    remove_stopwords=True,
    min_freq_to_keep=2
)

# 为整个数据集生成异构图
hetero_graph_output = to_hetero_graph.transform(corpus_docs, tokenizer)
print(f"hetero_graph_output type: {type(hetero_graph_output)}")
if isinstance(hetero_graph_output, list):
    if len(hetero_graph_output) > 0 and isinstance(hetero_graph_output[0], nx.Graph):
        hetero_graph = hetero_graph_output[0]
    elif len(hetero_graph_output) > 0 and isinstance(hetero_graph_output[0], dict):
        hetero_graph = hetero_graph_output[0].get('graph', None)
        if hetero_graph is None:
            raise ValueError("Heterogeneous graph not found in transform output.")
    else:
        raise TypeError("Unknown format for hetero_graph_output.")
elif isinstance(hetero_graph_output, dict):
    hetero_graph = hetero_graph_output.get('graph', None)
    if hetero_graph is None:
        raise ValueError("Heterogeneous graph not found in transform output.")
else:
    hetero_graph = hetero_graph_output  # 直接是图

hetero_adj, hetero_map = graph_to_scipy_sparse_matrix(hetero_graph, tokenizer)

hetero_adj = _normalize_adj(hetero_adj.tocsr()).tocoo()

hetero_adj = _scipy_to_torch(hetero_adj)


isg_graph_output = to_isg_graph.transform(corpus_docs, tokenizer)
print(f"isg_graph_output type: {type(isg_graph_output)}")
if isinstance(isg_graph_output, list):
    if len(isg_graph_output) > 0 and isinstance(isg_graph_output[0], nx.Graph):
        isg_graph = isg_graph_output[0]
    elif len(isg_graph_output) > 0 and isinstance(isg_graph_output[0], dict):
        isg_graph = isg_graph_output[0].get('graph', None)
        if isg_graph is None:
            raise ValueError("ISG graph not found in transform output.")
    else:
        raise TypeError("Unknown format for isg_graph_output.")
elif isinstance(isg_graph_output, dict):
    isg_graph = isg_graph_output.get('graph', None)
    if isg_graph is None:
        raise ValueError("ISG graph not found in transform output.")
else:
    isg_graph = isg_graph_output  # 直接是图

# 转换 ISG 图为 SciPy 稀疏矩阵和映射
isg_adj, isg_map = graph_to_scipy_sparse_matrix(isg_graph, tokenizer)
# 归一化并转换为 COO 格式
isg_adj = _normalize_adj(isg_adj.tocsr()).tocoo()
# 转换为 PyTorch 稀疏张量
isg_adj = _scipy_to_torch(isg_adj)

def validate_wgraph_id_maps(wgraph_id_to_tokenizer_id_maps):
    """
    验证每个 wgraph_id_to_tokenizer_id_map 是否具有从0到len(map)-1的连续键。

    参数:
        wgraph_id_to_tokenizer_id_maps (list of dict): 映射列表

    抛出:
        ValueError: 如果任何一个映射不满足键连续性要求
    """
    for idx, mapping in enumerate(wgraph_id_to_tokenizer_id_maps):
        if isinstance(mapping, list):
            for sub_mapping in mapping:
                if list(sub_mapping.keys()) != list(range(len(sub_mapping))):
                    raise ValueError(f"Mapping at index {idx} has incorrect keys")
        elif isinstance(mapping, dict):
            if list(mapping.keys()) != list(range(len(mapping))):
                raise ValueError(f"Mapping at index {idx} has incorrect keys")
    print("All maps have been validated.")

wgraph_id_to_tokenizer_id_maps = [hetero_map, isg_map]
validate_wgraph_id_maps(wgraph_id_to_tokenizer_id_maps)


train_messages, test_messages, train_labels, test_labels = train_test_split(
    messages, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = SpamDataset(
    train_messages, train_labels, tokenizer,
    word_graph_builder,
    hetero_adj, hetero_map,
    isg_adj, isg_map,
    max_length=128
)

test_dataset = SpamDataset(
    test_messages, test_labels, tokenizer,
    word_graph_builder,
    hetero_adj, hetero_map,
    isg_adj, isg_map,
    max_length=128
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=custom_collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate_fn
)


config = VGCNBertConfig.from_pretrained(
    '../bert-base-uncased',
    vgcn_hidden_dim=768,
    vgcn_graph_embds_dim=16,
    vgcn_hetero_graph_embds_dim=128,
    vgcn_isg_graph_embds_dim=128,
    sinusoidal_pos_embds=True,
    n_heads=12,
    n_layers=12,
    dim=768,
    hidden_dim=3072,
    dropout=0.1,
    attention_dropout=0.1,
    seq_classif_dropout=0.3,
    num_labels=2,
    vgcnbert_weight_init_mode="normal"
)
model = VGCNBertForSequenceClassification(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


optimizer = AdamW(model.parameters(), lr=1e-5)


from sklearn.utils.class_weight import compute_class_weight


class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

def train(model, dataloader, optimizer, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    scaler = GradScaler()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        cooccurrence_adjs = [adj.to(device) for adj in batch['cooccurrence_adjs']]
        cooccurrence_maps = batch['cooccurrence_maps']
        hetero_adj = [adj.to(device) for adj in batch['hetero_adj']]
        hetero_map = batch['hetero_map']
        isg_adj = [adj.to(device) for adj in batch['isg_adj']]
        isg_map = batch['isg_map']

        optimizer.zero_grad()
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cooccurrence_adjs=cooccurrence_adjs,
                cooccurrence_maps=cooccurrence_maps,
                hetero_adj=hetero_adj,
                hetero_map=hetero_map,
                isg_adj=isg_adj,
                isg_map=isg_map,
                output_attentions=False,
                output_hidden_states=False,
            )
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            torch.cuda.empty_cache()

        del outputs, loss, cooccurrence_adjs, cooccurrence_maps, hetero_adj, isg_adj, hetero_map, isg_map
        gc.collect()
        torch.cuda.empty_cache()

    average_loss = total_loss / len(dataloader)
    return average_loss


def evaluate(model, dataloader, device, epoch, roc_save_path='figure'):
    model.eval()
    total_correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    os.makedirs(roc_save_path, exist_ok=True)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            cooccurrence_adjs = [adj.to(device) for adj in batch['cooccurrence_adjs']]
            cooccurrence_maps = batch['cooccurrence_maps']
            hetero_adj = [adj.to(device) for adj in batch['hetero_adj']]
            hetero_map = batch['hetero_map']
            isg_adj = [adj.to(device) for adj in batch['isg_adj']]
            isg_map = batch['isg_map']

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cooccurrence_adjs=cooccurrence_adjs,
                    cooccurrence_maps=cooccurrence_maps,
                    hetero_adj=hetero_adj,
                    hetero_map=hetero_map,
                    isg_adj=isg_adj,
                    isg_map=isg_map,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())

            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)

            del outputs, logits, predictions, probabilities, cooccurrence_adjs, cooccurrence_maps, hetero_adj, isg_adj, hetero_map, isg_map
            gc.collect()
            torch.cuda.empty_cache()


    accuracy = total_correct / total

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, digits=4))


    if len(set(all_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Epoch {epoch + 1})')
        plt.legend(loc='lower right')

        # 生成文件名，并保存图像
        file_path = os.path.join(roc_save_path, f'roc_curve_dataset1_{epoch + 1}.png')
        plt.savefig(file_path)
        print(f"ROC curve saved as {file_path}")

        plt.close()
    else:
        print("只有一个类别存在，无法计算 AUC。")

    return accuracy

num_epochs = 8
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, device)
    val_accuracy = evaluate(model, test_loader, device, epoch)
    print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")
