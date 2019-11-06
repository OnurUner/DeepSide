import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from torch.autograd import Variable
import torch


class DrugDataset(Dataset):
    def __init__(self, dataset, output_size, drug_ids=None, exp_ids=None, 
                 adr_names=None, ontology_vectors=None, transform=None):
        self.dataset = dataset
        self.output_size = output_size
        self.drug_ids = drug_ids  # this will use only for random experiment sampler
        self.exp_ids = exp_ids  # this will use only to capture experiment ids at evaluation phase
        self.adr_names = adr_names
        self.transform = transform
        self.feature_ge = None
        self.feature_cs = None
        self.feature_meta = None
        self.feature_size = None
        if type(self.dataset[0][0]) is tuple:
            if len(self.dataset[0][0]) == 2:
                self.feature_ge = len(self.dataset[0][0][0])
                self.feature_cs = len(self.dataset[0][0][1])
                self.feature_size = self.feature_ge + self.feature_cs
            elif len(self.dataset[0][0]) == 3:
                self.feature_ge = len(self.dataset[0][0][0])
                self.feature_cs = len(self.dataset[0][0][1])
                self.feature_meta = len(self.dataset[0][0][2])
                self.feature_size = self.feature_ge + self.feature_cs + self.feature_meta
        elif self.transform is not None:
            self.feature_size = 0
        else:
            self.feature_size = len(self.dataset[0][0])

        if ontology_vectors is not None:
            self.ontology_vectors = ontology_vectors
        else:
            self.ontology_vectors = np.zeros((1))
        self.class_weights = self.class_weights()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index][0]
        if self.transform is not None:
            x = self.transform(x)
        y = self.dataset[index][1]
        return x, y

    def _print(self):
        print("Size:", len(self.dataset))
        print("Features:", self.feature_size)
        print("Output:", self.output_size)

    def class_weights(self):
        Y = []
        for i in range(len(self.dataset)):
            Y.append(self.dataset[i][1])
        Y = np.vstack(Y)

        neg_weights = []
        pos_weights = []
        for i in range(Y.shape[1]):
            w = compute_class_weight("balanced", np.unique(Y[:, i]), Y[:, i])
            neg_weights.append(w[0])
            pos_weights.append(w[1])
        return Variable(torch.from_numpy(np.array([neg_weights, pos_weights])).type(torch.FloatTensor),
                        requires_grad=False).cuda()