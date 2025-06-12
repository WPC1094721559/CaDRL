import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict
import logging


class DataProcessor:
    def __init__(self, rel2id_path, tokenizer_name="bert-base-cased",
                 max_seq_length=1024, max_entities=100):
        self.max_seq_length = max_seq_length
        self.max_entities = max_entities

        with open(rel2id_path, 'r', encoding='utf-8') as f:
            self.rel2id = json.load(f)
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.num_relations = len(self.rel2id)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.stats = {
            'total_docs': 0,
            'total_entities': 0,
            'total_relations': 0,
            'avg_doc_length': 0.0,
            'truncated_docs': 0
        }

        self.logger = logging.getLogger(__name__)

    def process_document(self, doc_data):
        try:
            title = doc_data.get("title", "")
            sents = doc_data["sents"]
            vertexSet = doc_data["vertexSet"]
            labels = doc_data.get("labels", [])

            doc_tokens, sent_map, sent_start_map = self._build_document_tokens(sents)
            input_ids, attention_mask, token_type_ids = self._tokenize_document(doc_tokens)
            entity_pos = self._process_entity_positions(vertexSet, sent_start_map, input_ids)
            hts = self._generate_entity_pairs(len(vertexSet))
            label_matrix = self._build_label_matrix(labels, hts, len(vertexSet))
            self._update_stats(doc_tokens, vertexSet, labels, input_ids)

            processed_data = {
                "title": title,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "entity_pos": entity_pos,
                "hts": hts,
                "labels": label_matrix,
                "num_entities": len(vertexSet),
                "num_relations": len(hts),
                "original_labels": labels,
                "sents": sents,
                "vertexSet": vertexSet
            }

            return processed_data

        except Exception as e:
            self.logger.error(f"处理文档失败 {doc_data.get('title', 'Unknown')}: {e}")
            return None

    def _build_document_tokens(self, sents):
        doc_tokens = []
        sent_map = []
        sent_start_map = []

        for sent_id, sent in enumerate(sents):
            sent_start_map.append(len(doc_tokens))
            for token in sent:
                doc_tokens.append(token)
                sent_map.append(sent_id)

        return doc_tokens, sent_map, sent_start_map

    def _tokenize_document(self, doc_tokens):
        text = " ".join(doc_tokens)
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0).tolist()
        attention_mask = encoded['attention_mask'].squeeze(0).tolist()
        token_type_ids = encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(0).tolist()

        return input_ids, attention_mask, token_type_ids

    def _process_entity_positions(self, vertexSet, sent_start_map, input_ids):
        entity_pos = []
        offset = 1

        for entity_mentions in vertexSet:
            entity_positions = []

            for mention in entity_mentions:
                sent_id = mention["sent_id"]
                start_pos = mention["pos"][0]
                end_pos = mention["pos"][1]

                if sent_id < len(sent_start_map):
                    doc_start = sent_start_map[sent_id] + start_pos
                    doc_end = sent_start_map[sent_id] + end_pos
                    adj_start = min(doc_start + offset, len(input_ids) - 2)
                    adj_end = min(doc_end + offset - 1, len(input_ids) - 2)
                    if adj_start < adj_end:
                        entity_positions.append([adj_start, adj_end])
            if not entity_positions:
                entity_positions.append([1, 1])
            entity_pos.append(entity_positions)

        return entity_pos

    def _generate_entity_pairs(self, num_entities):
        hts = []
        for h in range(num_entities):
            for t in range(num_entities):
                if h != t:
                    hts.append([h, t])
        return hts

    def _build_label_matrix(self, labels, hts, num_entities):
        label_matrix = []

        label_map = {}
        for label in labels:
            key = (label["h"], label["t"])
            if key not in label_map:
                label_map[key] = []
            label_map[key].append(self.rel2id.get(label["r"], 0))

        for h, t in hts:
            label_vector = [0] * self.num_relations
            if (h, t) in label_map:
                for rel_id in label_map[(h, t)]:
                    if rel_id < self.num_relations:
                        label_vector[rel_id] = 1
            if sum(label_vector) == 0:
                label_vector[0] = 1
            label_matrix.append(label_vector)

        return label_matrix

    def _update_stats(self, doc_tokens, vertexSet, labels, input_ids):
        self.stats['total_docs'] += 1
        self.stats['total_entities'] += len(vertexSet)
        self.stats['total_relations'] += len(labels)
        doc_length = len(doc_tokens)
        if self.stats['total_docs'] == 1:
            self.stats['avg_doc_length'] = doc_length
        else:
            self.stats['avg_doc_length'] = (self.stats['avg_doc_length'] * (self.stats['total_docs'] - 1)
                                            + doc_length) / self.stats['total_docs']

        if len(doc_tokens) + 2 > self.max_seq_length:
            self.stats['truncated_docs'] += 1

    def get_statistics(self):
        if self.stats['total_docs'] > 0:
            return {
                'total_docs': self.stats['total_docs'],
                'total_entities': self.stats['total_entities'],
                'total_relations': self.stats['total_relations'],
                'avg_entities_per_doc': self.stats['total_entities'] / self.stats['total_docs'],
                'avg_relations_per_doc': self.stats['total_relations'] / self.stats['total_docs'],
                'avg_doc_length': self.stats['avg_doc_length'],
                'truncation_rate': self.stats['truncated_docs'] / self.stats['total_docs'],
                'num_relation_types': self.num_relations
            }
        return self.stats


class DocREDDataset(Dataset):
    def __init__(self, data_path, rel2id_path, tokenizer_name="..PLM/bert-base-cased",
                 max_seq_length=1024, mode="train"):
        self.mode = mode
        self.data_path = data_path

        self.processor = DataProcessor(
            rel2id_path, tokenizer_name, max_seq_length
        )
        # self.logger = logging.getLogger(__name__)
        # self.logger.info(f"加载{mode}数据: {data_path}")
        self.data = self._load_and_process_data(data_path)
        #
        # stats = self.processor.get_statistics()
        # for key, value in stats.items():
        #     self.logger.info(f"  {key}: {value}")

    def _load_and_process_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        processed_data = []
        failed_count = 0

        for doc_data in raw_data:
            processed = self.processor.process_document(doc_data)
            if processed is not None:
                processed_data.append(processed)
            else:
                failed_count += 1
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_relation_mapping(self):
        return self.processor.rel2id, self.processor.id2rel


class AdjacencyMatrixBuilder:
    def __init__(self, n_entities, n_relations):
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.logger = logging.getLogger(__name__)

    def build_from_training_data(self, dataset):
        adjacency_matrices = torch.zeros(self.n_relations, self.n_entities, self.n_entities)
        relation_count = defaultdict(int)

        for doc_data in dataset:
            original_labels = doc_data.get('original_labels', [])

            for label in original_labels:
                h = label['h']
                t = label['t']
                r = dataset.processor.rel2id.get(label['r'], 0)
                if h < self.n_entities and t < self.n_entities and r < self.n_relations:
                    adjacency_matrices[r, h, t] = 1.0
                    relation_count[r] += 1
        total_edges = sum(relation_count.values())

        top_relations = sorted(relation_count.items(), key=lambda x: x[1], reverse=True)[:10]
        for rel_id, count in top_relations:
            rel_name = dataset.processor.id2rel.get(rel_id, f"REL_{rel_id}")

        return adjacency_matrices

    def build_from_knowledge_graph(self, kg_file):
        adjacency_matrices = torch.zeros(self.n_relations, self.n_entities, self.n_entities)

        with open(kg_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = parts
                    try:
                        h_id, r_id, t_id = int(h), int(r), int(t)
                        if (h_id < self.n_entities and t_id < self.n_entities and
                                r_id < self.n_relations):
                            adjacency_matrices[r_id, h_id, t_id] = 1.0
                    except ValueError:
                        continue

        return adjacency_matrices


def collate_fn(batch):
    max_len = max([len(item["input_ids"]) for item in batch])

    collated_batch = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'entity_pos': [],
        'hts': [],
        'labels': [],
        'titles': []
    }

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        collated_batch['input_ids'].append(
            item["input_ids"] + [0] * pad_len
        )
        collated_batch['attention_mask'].append(
            item["attention_mask"] + [0] * pad_len
        )
        collated_batch['token_type_ids'].append(
            item["token_type_ids"] + [0] * pad_len
        )

        collated_batch['entity_pos'].append(item["entity_pos"])
        collated_batch['hts'].append(item["hts"])
        collated_batch['labels'].append(item["labels"])
        collated_batch['titles'].append(item["title"])

    collated_batch['input_ids'] = torch.tensor(collated_batch['input_ids'], dtype=torch.long)
    collated_batch['attention_mask'] = torch.tensor(collated_batch['attention_mask'], dtype=torch.float)
    collated_batch['token_type_ids'] = torch.tensor(collated_batch['token_type_ids'], dtype=torch.long)

    return collated_batch
