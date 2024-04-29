import torch
import os
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('-' * 10, string, '-' * 10)
    print('Average interactions:', X.sum(1).mean(0).item())
    nonzero_row_indices, nonzero_col_indices = X.nonzero()
    unique_nonzero_row_indices = np.unique(nonzero_row_indices)
    unique_nonzero_col_indices = np.unique(nonzero_col_indices)
    print('Non-zero rows:', len(unique_nonzero_row_indices) / X.shape[0])
    print('Non-zero columns:', len(unique_nonzero_col_indices) / X.shape[1])
    print('Matrix density:', len(nonzero_row_indices) / (X.shape[0] * X.shape[1]))


class TrainDataset(Dataset):
    def __init__(self, learner_course_pairs, learner_course_graph, num_courses, neg_sample=1):
        self.learner_course_pairs = learner_course_pairs
        self.learner_course_graph = learner_course_graph
        self.num_courses = num_courses
        self.neg_sample = neg_sample

    def __getitem__(self, index):
        learner, positive_course = self.learner_course_pairs[index]
        pos_and_neg_courses = [positive_course]

        while True:
            i = np.random.randint(self.num_courses)
            if self.learner_course_graph[learner, i] == 0 and i not in pos_and_neg_courses:
                pos_and_neg_courses.append(i)
                if len(pos_and_neg_courses) == self.neg_sample + 1:
                    break

        return torch.LongTensor([learner]), torch.LongTensor(pos_and_neg_courses)

    def __len__(self):
        return len(self.learner_course_pairs)


class TestDataset(Dataset):
    def __init__(self, learner_course_graph, learner_course_graph_train):
        self.learner_course_graph = learner_course_graph
        self.masked_learner_course = learner_course_graph_train

    def __getitem__(self, index):
        learner_course_ground_truth = torch.from_numpy(self.learner_course_graph[index].toarray()).squeeze()
        learner_course_mask = torch.from_numpy(self.masked_learner_course[index].toarray()).squeeze()
        return index, learner_course_ground_truth, learner_course_mask

    def __len__(self):
        return self.learner_course_graph.shape[0]


class Datasets:
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_learners, self.num_courses, self.num_concepts = self.get_data_size()

        learner_course_pairs_train, learner_course_graph_train = self.get_learner_course("train")
        learner_course_pairs_val, learner_course_graph_val = self.get_learner_course("tune")
        learner_course_pairs_test, learner_course_graph_test = self.get_learner_course("test")
        learner_concept_pairs, learner_concept_graph = self.get_learner_concept()
        course_concept_graph = self.get_course_concept()

        self.course_train_data = TrainDataset(learner_course_pairs_train, learner_course_graph_train,
                                              self.num_courses, conf["neg_num"])
        self.course_val_data = TestDataset(learner_course_graph_val, learner_course_graph_train)
        self.course_test_data = TestDataset(learner_course_graph_test, learner_course_graph_train)

        self.graphs = [learner_course_graph_train, learner_concept_graph, course_concept_graph]

        self.train_loader = DataLoader(self.course_train_data, batch_size=batch_size_train, shuffle=True,
                                       num_workers=12, drop_last=True)
        self.val_loader = DataLoader(self.course_val_data, batch_size=batch_size_test, shuffle=False, num_workers=12)
        self.test_loader = DataLoader(self.course_test_data, batch_size=batch_size_test, shuffle=False, num_workers=12)

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]

    def get_learner_course(self, task):
        with open(os.path.join(self.path, self.name, 'learner_course_{}.txt'.format(task)), 'r') as f:
            learner_course_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(learner_course_pairs, dtype=np.int32)
        values = np.ones(len(learner_course_pairs), dtype=np.float32)
        learner_course_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])),
                                             shape=(self.num_learners, self.num_courses)).tocsr()
        print_statistics(learner_course_graph, "#Learner-Course Statistics in %s" % task)
        return learner_course_pairs, learner_course_graph

    def get_learner_concept(self):
        with open(os.path.join(self.path, self.name, 'learner_concept.txt'), 'r') as f:
            learner_concept_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(learner_concept_pairs, dtype=np.int32)
        values = np.ones(len(learner_concept_pairs), dtype=np.float32)
        learner_concept_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])),
                                              shape=(self.num_learners, self.num_concepts)).tocsr()
        print_statistics(learner_concept_graph, '#Learner-Concept Statistics')
        return learner_concept_pairs, learner_concept_graph

    def get_course_concept(self):
        with open(os.path.join(self.path, self.name, 'course_concept.txt'), 'r') as f:
            course_concept_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(course_concept_pairs, dtype=np.int32)
        values = np.ones(len(course_concept_pairs), dtype=np.float32)
        course_concept_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])),
                                             shape=(self.num_courses, self.num_concepts)).tocsr()
        print_statistics(course_concept_graph, '#Course-Concept Statistics')
        return course_concept_graph
