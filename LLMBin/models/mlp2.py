import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU(),
                 pretrained_model=None, cov_model=None, covmodel_notl2normalize=False,
                 llm_model=None, llmmodel_notl2normalize=False,):
        super(EmbeddingNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.cov_model = cov_model
        self.llm_model = llm_model
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        self.covmodel_notl2normalize = covmodel_notl2normalize
        self.llmmodel_notl2normalize = llmmodel_notl2normalize
        if ps == 0:
            ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]), actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(
                n_in=emb_szs[i], n_out=emb_szs[i + 1], bn=use_bn, p=ps[i], actn=actn
            )
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)
        project_layer= [actn, nn.Linear(self.out_sz,self.out_sz)]
        # project_layer= [nn.Linear(self.out_sz,self.out_sz),actn,nn.Linear(self.out_sz,self.out_sz)]
        self.fc2 = nn.Sequential(*project_layer)

    def bn_drop_lin(
            self,
            n_in: int,
            n_out: int,
            bn: bool = True,
            p: float = 0.0,
            actn: nn.Module = None,
    ):
        # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/layers.py#L44
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, kmer_feat, cov_feat=None, llm_feat=None):
        if self.pretrained_model is not None:
            # kmeremb = self.pretrained_model(kmer_feat)
            kmer_feat_emb = F.normalize(self.pretrained_model(kmer_feat))
        else:
            kmer_feat_emb = kmer_feat

        if self.cov_model is not None:
            if self.covmodel_notl2normalize:
                cov_feat_emb = self.cov_model(cov_feat)
            else:
                cov_feat_emb = F.normalize(self.cov_model(cov_feat))
        else:
            cov_feat_emb = cov_feat

        if self.llm_model is not None:
            if self.llmmodel_notl2normalize:
                llm_feat_emb = self.llm_model(llm_feat)
            else:
                llm_feat_emb = F.normalize(self.llm_model(llm_feat))
        else:
            llm_feat_emb = llm_feat

        x = torch.cat([kmer_feat_emb, cov_feat_emb, llm_feat_emb], dim=-1)
        output = self.fc(x)

        return output, kmer_feat_emb, cov_feat_emb, llm_feat_emb #F.normalize(output) #output #/ torch.linalg.vector_norm(output, dim=-1, keepdim=True)
