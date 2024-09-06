import os
import dgl
import torch
import pandas as pd
import numpy as np
from dgl.data import DGLDataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from dgl.data.utils import save_graphs, load_graphs, generate_mask_tensor, idx2mask
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold


class PubMedDataset(DGLDataset):
    def __init__(
        self, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=False
    ):
        self.graph = HeteroData()
        self._ntypes = {"G": "Gene", "D": "Disease", "C": "Chemical", "S": "Species"}
        self._relations = [
            ("Gene", "Gene-Gene", "Gene"),
            ("Gene", "Gene-Disease", "Disease"),
            ("Disease", "Disease-Disease", "Disease"),
            ("Chemical", "Chemical-Gene", "Gene"),
            ("Chemical", "Chemical-Disease", "Disease"),
            ("Chemical", "Chemical-Chemical", "Chemical"),
            ("Chemical", "Chemical-Species", "Species"),
            ("Species", "Species-Gene", "Gene"),
            ("Species", "Species-Disease", "Disease"),
            ("Species", "Species-Species", "Species"),
        ]
        self._classes = [
            "cardiovascular_disease",
            "glandular_disease",
            "nervous_disorder",
            "communicable_disease",
            "inflammatory_disease",
            "pycnosis",
            "skin_disease",
            "cancer",
        ]
        curr_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(curr_dir)
        self.data_path = os.path.join(parent_dir, "../data/CKD_data/PubMed")
        self.add_reverse_relation()
        super(PubMedDataset, self).__init__(
            name="pubmed",
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def add_reverse_relation(self):
        for rel_tuple in self._relations:
            src, rel, dst = rel_tuple
            rev_rel = (dst, f"{dst}-{src}", src)
            if rev_rel not in self._relations:
                self._relations.append(rev_rel)

    def download(self):
        pass

    def load(self):
        print("loading graph")
        graphs, _ = load_graphs(os.path.join(self.data_path, "PubMed_dgl_graph.bin"))
        self.graph = graphs[0]

    def save(self):
        print("saving graph")
        save_graphs(os.path.join(self.data_path, "PubMed_dgl_graph.bin"), [self.graph])

    def process(self):
        nodes_file = pd.read_csv(
            os.path.join(self.data_path, "node.dat"),
            sep="\t",
            names=["node_id", "node_name", "node_type", "node_attributes"],
        )
        nodes = nodes_file["node_id"].tolist()
        nodes_type = nodes_file["node_type"].tolist()
        nodes_attributes = nodes_file["node_attributes"].tolist()

        # ## add node type to link.test.dat
        # link_test_file = pd.read_csv(
        #     os.path.join(self.data_path, "link.dat.test"),
        #     sep="\t",
        #     names=["u_id", "v_id", "link_type"],
        # )
        # u_nodes = link_test_file["u_id"].tolist()
        # v_nodes = link_test_file["v_id"].tolist()
        # u_nodes_ntypes = []
        # v_nodes_ntypes = []
        # link_type = link_test_file["link_type"].tolist()
        # for u, v, t in zip(u_nodes, v_nodes, link_type):
        #     u_nodes_ntypes.append(nodes_type[nodes.index(u)])
        #     v_nodes_ntypes.append(nodes_type[nodes.index(v)])
        # link_test_transform = pd.DataFrame(
        #     {"u_id": u_nodes, "u_type": u_nodes_ntypes, "v_id": v_nodes, "v_type": v_nodes_ntypes, "link_type": link_type}
        # )
        # link_test_transform.to_csv(os.path.join(self.data_path, "link.dat.test.trans"), sep="\t", index=False)
        ## mapping each node id(specific type) into new id by dictionary
        node_dict = {ntype: {"id": [], "feat": []} for ntype in self._ntypes.values()}
        for n, t, attr in zip(nodes, nodes_type, nodes_attributes):

            ntype = list(self._ntypes.values())[t]

            node_dict[ntype]["id"].append(n)
            feat = attr.split(",")
            feat = [float(f) for f in feat]
            node_dict[ntype]["feat"].append(feat)

        ## sort node id and feature by node id
        for ntype in self._ntypes.values():
            combined_list = zip(node_dict[ntype]["id"], node_dict[ntype]["feat"])

            sorted_combined_list = sorted(combined_list, key=lambda x: x[0])
            sort_id = [x[0] for x in sorted_combined_list]
            sort_feat = [x[1] for x in sorted_combined_list]

            node_dict[ntype]["id"] = sort_id
            node_dict[ntype]["feat"] = torch.tensor(sort_feat)
        np.save(os.path.join(self.data_path, "target_node_dict.npy"), np.array(node_dict[self.predict_ntype]["id"]))
        self._read_edges(node_dict)

        self._read_feats_labels(node_dict)

    def _read_edges(self, node_dict):
        edges = {}
        edges_file = pd.read_csv(
            os.path.join(self.data_path, "link.dat"),
            sep="\t",
            names=["u_id", "v_id", "link_type", "link_weight"],
        )

        src = edges_file["u_id"].tolist()
        dst = edges_file["v_id"].tolist()
        etype = edges_file["link_type"].tolist()
        for s, d, t in tqdm(zip(src, dst, etype), total=len(src)):
            rel_tuple = self._relations[int(t)]
            src_t, rel, dst_t = rel_tuple
            map_s_id = node_dict[src_t]["id"].index(s)
            map_d_id = node_dict[dst_t]["id"].index(d)
            if rel_tuple not in edges:
                edges[rel_tuple] = ([map_s_id], [map_d_id])
            else:
                edges[rel_tuple][0].append(map_s_id)
                edges[rel_tuple][1].append(map_d_id)
            if src_t != dst_t:
                ## self-defined rev relation
                rev_rel = (dst_t, f"{dst_t}-{src_t}", src_t)

                if rev_rel not in edges:
                    edges[rev_rel] = ([map_d_id], [map_s_id])
                else:
                    edges[rev_rel][0].append(map_d_id)
                    edges[rev_rel][1].append(map_s_id)
        self.graph = dgl.heterograph(edges)

        for ntype in self._ntypes.values():
            ntype_num_nodes = self.graph.num_nodes(ntype)
            ntype_nei_degree = {ntype: torch.zeros(ntype_num_nodes) for ntype in self._ntypes.values()}
            use_relations = [rel_tuple for rel_tuple in self._relations if rel_tuple[2] == ntype]
            for rel_tuple in use_relations:
                src, rel, dst = rel_tuple
                ntype_nei_degree[src] = self.graph.in_degrees(torch.arange(ntype_num_nodes), etype=rel)
            self.graph.nodes[ntype].data["in_degree"] = torch.stack([ntype_nei_degree[ntype] for ntype in self._ntypes.values()], dim=1)
            # print(self.g.nodes[ntype].data["in_degree"].shape)

    def _read_feats_labels(self, node_dict):
        split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
        label_train_file = pd.read_csv(
            os.path.join(self.data_path, "label.dat"),
            sep="\t",
            names=["node_id", "node_name", "node_type", "node_label"],
        )
        label_test_file = pd.read_csv(
            os.path.join(self.data_path, "label.dat.test"),
            sep="\t",
            names=["node_id", "node_name", "node_type", "node_label"],
        )
        label_file = pd.concat([label_train_file, label_test_file], ignore_index=True)
        ## assign node feature
        for ntype in self._ntypes.values():
            self.graph.nodes[ntype].data["feat"] = node_dict[ntype]["feat"]

        ## assign pred node label in graph
        pred_num_nodes = self.graph.num_nodes(self.predict_ntype)
        self.graph.nodes[self.predict_ntype].data["label"] = torch.full((pred_num_nodes, 1), -1)
        label_nodes_indices = []
        for _, row in label_file.iterrows():
            node_id = row["node_id"]
            mapped_node_id = node_dict[self.predict_ntype]["id"].index(node_id)
            # node_type = row["node_type"]
            node_label = row["node_label"]
            self.graph.nodes[self.predict_ntype].data["label"][mapped_node_id] = torch.tensor([node_label])
            label_nodes_indices.append(mapped_node_id)

        ## split label node into train valid test

        ## self-define split ratio
        # num_train_nodes = int(len(label_nodes_indices) * split_ratio["train"])
        # num_valid_nodes = int(len(label_nodes_indices) * split_ratio["val"])
        # eva_indices = {
        #     "train": label_nodes_indices[:num_train_nodes],
        #     "val": label_nodes_indices[num_train_nodes : num_train_nodes + num_valid_nodes],
        #     "test": label_nodes_indices[num_train_nodes + num_valid_nodes :],
        # }
        label_nodes_indices = np.array(label_nodes_indices)
        mask = generate_mask_tensor(idx2mask(label_nodes_indices, pred_num_nodes))

        self.graph.nodes[self.predict_ntype].data["total"] = torch.tensor(mask)

    def has_cache(self):
        # return False
        return os.path.exists(os.path.join(self.data_path, "PubMed_dgl_graph.bin"))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def relations(self):
        return self._relations

    @property
    def predict_ntype(self):
        return "Disease"

    @property
    def has_label_ratio(self):
        return False

    @property
    def multilabel(self):
        return False
