import datetime
import sys
import warnings
import numpy as np

import torch
from torch_geometric.nn.models import MetaPath2Vec
#from dgl.nn.pytorch import MetaPath2Vec
from hgmae.models.edcoder import PreModel
from hgmae.utils import evaluate, evaluate_cluster, load_best_configs, load_data, metapath2vec_train, preprocess_features, set_random_seed,LGS_node_classification_evaluate
from hgmae.utils.params import build_args
from hgmae.utils.preprocess_Freebase import FreebaseDataset
from hgmae.utils.preprocess_PubMed import PubMedDataset 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import dgl
import scipy.sparse as sp
from tqdm import tqdm
warnings.filterwarnings("ignore")


def visualization(embs, labels, dataset, display=False):
    perplexity = min(30, embs.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embs_2d = tsne.fit_transform(embs)
    if display:
        plt.figure(figsize=(12, 8))
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "pink", "brown", "gray"]
        for label in np.unique(labels):
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            plt.scatter(embs_2d[indices, 0], embs_2d[indices, 1], color=colors[label], label=f"Class {label}", alpha=0.6)

        plt.title("t-SNE visualization of node embeddings with class labels")
        plt.xlabel("x t-SNE vector")
        plt.ylabel("y t-SNE vector")
        plt.legend()
        plt.savefig(f"{dataset}_{datetime.datetime.now()}.png")
    return embs_2d


def main(args):
    # random seed
    set_random_seed(args.seed)
    if args.dataset == "PubMed" or args.dataset == "Freebase":
        (nei_index, feats, mps, pos,label,label_indices), g, processed_metapaths = load_data(args.dataset, args.ratio, args.type_num)
    else:
        (nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test), g, processed_metapaths = load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    
    feats_dim_list = [i.shape[1] for i in feats]
    num_mp = int(len(mps))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", num_mp)
    print(processed_metapaths)
    print(g.edge_index_dict)
    if args.use_mp2vec_feat_pred:
        assert args.mps_embedding_dim > 0
        metapath_model = MetaPath2Vec(
            g.edge_index_dict,
            args.mps_embedding_dim,
            processed_metapaths,
            args.mps_walk_length,
            args.mps_context_size,
            args.mps_walks_per_node,
            args.mps_num_negative_samples,
            sparse=True,
        )
        metapath2vec_train(args, metapath_model, args.mps_epoch, args.device)
        mp2vec_feat = metapath_model("target").detach()

        # free up memory
        del metapath_model
        if args.device.type == "cuda":
            mp2vec_feat = mp2vec_feat.cpu()
            torch.cuda.empty_cache()
        mp2vec_feat = torch.FloatTensor(preprocess_features(mp2vec_feat))
        feats[0] = torch.hstack([feats[0], mp2vec_feat])

    # model
    focused_feature_dim = feats_dim_list[0]
    model = PreModel(args, num_mp, focused_feature_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    # scheduler
    if args.scheduler:
        print("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    model.to(args.device)
    feats = [feat.to(args.device) for feat in feats]
    mps = [mp.to(args.device) for mp in mps]
    label = label.to(args.device)
    if args.dataset != "PubMed" and args.dataset != "Freebase":
        idx_train = [i.to(args.device) for i in idx_train]
        idx_val = [i.to(args.device) for i in idx_val]
        idx_test = [i.to(args.device) for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    best_model_state_dict = None
    for epoch in range(args.mae_epochs):
        model.train()
        optimizer.zero_grad()
        loss, loss_item = model(feats, mps, nei_index=nei_index, epoch=epoch)
        print(f"Epoch: {epoch}, loss: {loss_item}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            best_model_state_dict = model.state_dict()
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("Early stopping!")
            break
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        #break
    print("The best epoch is: ", best_t)
    model.load_state_dict(best_model_state_dict)
    model.eval()
    embeds = model.get_embeds(feats, mps, nei_index)
    if args.task == "classification":
        macro_score_list, micro_score_list, auc_score_list = [], [], []
        if(args.dataset == "PubMed" or args.dataset == "Freebase"):
            mean,std=LGS_node_classification_evaluate(embeds,label,label_indices)
            print(
                "\t ACC:[{:4f},{:4f}] Micro-F1:[{:.4f}, {:.4f}] Macro-F1:[{:.4f}, {:.4f}]  \n".format(
                    mean["auc_roc"],
                    std["auc_roc"],
                    mean["micro_f1"],
                    std["micro_f1"],
                    mean["macro_f1"],
                    std["macro_f1"],
                )
            )
        else:
            for i in range(len(idx_train)):
                macro_score, micro_score, auc_score = evaluate(
                    embeds, idx_train[i], idx_val[i], idx_test[i], label, nb_classes, args.device, args.eva_lr, args.eva_wd
                )
                macro_score_list.append(macro_score)
                micro_score_list.append(micro_score)
                auc_score_list.append(auc_score)
                
    elif args.task == "clustering" or epoch:
        # node clustering
        nmi_list, ari_list = [], []

        embeds = embeds.cpu().data.numpy()
        label = np.argmax(label.cpu().data.numpy(), axis=-1)
        embeds_2d = visualization(embeds, label, args.dataset, False)

        for kmeans_random_state in range(10):
            nmi, ari = evaluate_cluster(embeds, label, args.n_labels, kmeans_random_state)
            nmi_list.append(nmi)
            ari_list.append(ari)
        print(
            "\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]".format(
                np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)
            )
        )

    else:
        sys.exit("wrong args.task.")

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")


if __name__ == "__main__":
    args = build_args()
    if torch.cuda.is_available():
        print(args.gpu)
        args.device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        args.device = torch.device("cpu")

    if args.use_cfg:
        if args.task == "classification":
            config_file_name = "configs.yml"
        elif args.task == "clustering":
            config_file_name = "clustering_configs.yml"
        else:
            sys.exit(f"No available config file for task: {args.task}")
        args = load_best_configs(args, config_file_name)
    print(args)
    print(args.use_mp2vec_feat_pred)
    main(args)
