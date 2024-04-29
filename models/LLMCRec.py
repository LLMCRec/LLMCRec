import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.sparse as sp


def matrix_to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def laplace_transform(graph):
    sqrt_row_sum = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    sqrt_col_sum = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = sqrt_row_sum @ graph @ sqrt_col_sum
    return graph


def get_bpr_loss(pred):
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)
    loss = - torch.log(torch.sigmoid(pos - negs))
    loss = torch.mean(loss)
    return loss


def random_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio])
    values = mask * values
    return values


class LLMCRec(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.embedding_size = conf["embedding_size"]
        self.num_learners = conf["num_learners"]
        self.num_courses = conf["num_courses"]
        self.num_concepts = conf["num_concepts"]

        self.init_emb_KG()

        self.learner_course_graph, self.learner_concept_graph, self.course_concept_graph = raw_graph

        self.get_original_concept_grained_graph()
        self.get_original_course_grained_graph()
        self.get_original_course_concept_graph()

        self.get_concept_grained_graph()
        self.get_course_grained_graph()
        self.get_course_concept_graph()

        self.init_md()

        self.layer_num = self.conf["layer_num"]
        self.c_temp = self.conf["c_temp"]

    def init_md(self):
        self.course_grained_dropout = nn.Dropout(self.conf["course_grained_ratio"], True)
        self.concept_grained_dropout = nn.Dropout(self.conf["concept_grained_ratio"], True)
        self.course_concept_dropout = nn.Dropout(self.conf["course_concept_ratio"], True)

    def init_emb_KG(self):
        df = pd.read_csv("./datasets/" + self.conf["dataset"] + "/ent-dismult-5-128.csv", header=None)
        id_map = pd.read_csv("./datasets/" + self.conf["dataset"] + "/rs2kgid.txt", sep='\t', header=None)

        self.learners_feature = torch.FloatTensor(self.num_learners, self.embedding_size)
        nn.init.xavier_normal_(self.learners_feature)

        self.courses_feature = torch.FloatTensor(self.num_courses, self.embedding_size)
        for i in range(self.num_courses):
            if "c_" + str(i) in id_map[0].values:
                self.courses_feature[i] = torch.FloatTensor(df.iloc[id_map[id_map[0] == "c_" + str(i)][1].values[0]])
            else:
                nn.init.xavier_normal_(self.courses_feature[i].unsqueeze(0))

        self.concepts_feature = torch.FloatTensor(self.num_concepts, self.embedding_size)
        for i in range(self.num_concepts):
            if "k_" + str(i) in id_map[0].values:
                self.concepts_feature[i] = torch.FloatTensor(df.iloc[id_map[id_map[0] == "k_" + str(i)][1].values[0]])
            else:
                nn.init.xavier_normal_(self.concepts_feature[i].unsqueeze(0))
        linear_layer_1 = nn.Linear(self.embedding_size, 64)
        linear_layer_2 = nn.Linear(64, 64)
        relu_activation = nn.ReLU()
        model = nn.Sequential(
            linear_layer_1,
            linear_layer_2,
            relu_activation
        )
        self.learners_feature = model(self.learners_feature)
        self.courses_feature = model(self.courses_feature)
        self.concepts_feature = model(self.concepts_feature)
        self.learners_feature = nn.Parameter(self.learners_feature)
        self.courses_feature = nn.Parameter(self.courses_feature)
        self.concepts_feature = nn.Parameter(self.concepts_feature)

    def get_course_grained_graph(self):
        learner_course_graph = self.learner_course_graph
        device = self.device
        dp_ratio = self.conf["course_grained_ratio"]

        course_grained_graph = sp.bmat(
            [[sp.csr_matrix((learner_course_graph.shape[0], learner_course_graph.shape[0])), learner_course_graph],
             [learner_course_graph.T, sp.csr_matrix((learner_course_graph.shape[1], learner_course_graph.shape[1]))]])

        if dp_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = course_grained_graph.tocoo()
                values = random_edge_dropout(graph.data, dp_ratio)
                course_grained_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.course_grained_graph = matrix_to_tensor(laplace_transform(course_grained_graph)).to(device)

    def get_original_course_grained_graph(self):
        learner_course_graph = self.learner_course_graph
        device = self.device
        course_level_graph = sp.bmat(
            [[sp.csr_matrix((learner_course_graph.shape[0], learner_course_graph.shape[0])), learner_course_graph],
             [learner_course_graph.T, sp.csr_matrix((learner_course_graph.shape[1], learner_course_graph.shape[1]))]])
        self.course_grained_graph_ori = matrix_to_tensor(laplace_transform(course_level_graph)).to(device)

    def get_concept_grained_graph(self):
        learner_concept_graph = self.learner_concept_graph
        device = self.device
        dp_ratio = self.conf["concept_grained_ratio"]

        concept_grained_graph = sp.bmat(
            [[sp.csr_matrix((learner_concept_graph.shape[0], learner_concept_graph.shape[0])), learner_concept_graph],
             [learner_concept_graph.T,
              sp.csr_matrix((learner_concept_graph.shape[1], learner_concept_graph.shape[1]))]])
        if dp_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = concept_grained_graph.tocoo()
                values = random_edge_dropout(graph.data, dp_ratio)
                concept_grained_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.concept_grained_graph = matrix_to_tensor(laplace_transform(concept_grained_graph)).to(device)

    def get_original_concept_grained_graph(self):
        learner_concept_graph = self.learner_concept_graph
        device = self.device
        concept_grained_graph = sp.bmat(
            [[sp.csr_matrix((learner_concept_graph.shape[0], learner_concept_graph.shape[0])), learner_concept_graph],
             [learner_concept_graph.T,
              sp.csr_matrix((learner_concept_graph.shape[1], learner_concept_graph.shape[1]))]])
        self.concept_grained_graph_ori = matrix_to_tensor(laplace_transform(concept_grained_graph)).to(device)

    def get_course_concept_graph(self):
        course_concept_graph = self.course_concept_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            dp_ratio = self.conf["course_concept_ratio"]
            graph = self.course_concept_graph.tocoo()
            values = random_edge_dropout(graph.data, dp_ratio)
            course_concept_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        course_size = course_concept_graph.sum(axis=1) + 1e-8
        course_concept_graph = sp.diags(1 / course_size.A.ravel()) @ course_concept_graph
        self.course_aggregate_graph = matrix_to_tensor(course_concept_graph).to(device)

    def get_original_course_concept_graph(self):
        course_concept_graph = self.course_concept_graph
        device = self.device
        course_size = course_concept_graph.sum(axis=1) + 1e-8
        course_concept_graph = sp.diags(1 / course_size.A.ravel()) @ course_concept_graph
        self.course_aggregate_graph_ori = matrix_to_tensor(course_concept_graph).to(device)

    def graph_propagate(self, graph, A_feature, B_feature, md, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.layer_num):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                features = md(features)
            features = features / (i + 2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def get_concept_grained_courses_representations(self, concept_grained_concept_feature, test):
        if test:
            concept_grained_course_feature = torch.matmul(self.course_aggregate_graph_ori,
                                                          concept_grained_concept_feature)
        else:
            concept_grained_course_feature = torch.matmul(self.course_aggregate_graph, concept_grained_concept_feature)
        if self.conf["course_concept_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            concept_grained_course_feature = self.course_concept_dropout(concept_grained_course_feature)
        return concept_grained_course_feature

    def propagate(self, test=False):
        if test:
            course_grained_learners_feature, course_grained_courses_feature = self.graph_propagate(
                self.course_grained_graph_ori,
                self.learners_feature, self.courses_feature,
                self.course_grained_dropout, test)
        else:
            course_grained_learners_feature, course_grained_courses_feature = self.graph_propagate(
                self.course_grained_graph,
                self.learners_feature, self.courses_feature,
                self.course_grained_dropout, test)

        if test:
            concept_grained_learners_feature, concept_grained_concepts_feature = self.graph_propagate(
                self.concept_grained_graph_ori,
                self.learners_feature, self.concepts_feature,
                self.concept_grained_dropout, test)
        else:
            concept_grained_learners_feature, concept_grained_concepts_feature = self.graph_propagate(
                self.concept_grained_graph, self.learners_feature,
                self.concepts_feature,
                self.concept_grained_dropout, test)

        concept_grained_course_feature = self.get_concept_grained_courses_representations(
            concept_grained_concepts_feature, test)

        learners_feature = [concept_grained_learners_feature, course_grained_learners_feature]
        courses_feature = [concept_grained_course_feature, course_grained_courses_feature]

        return learners_feature, courses_feature

    def get_contrastive_loss(self, A, B):
        A = A[:, 0, :]
        B = B[:, 0, :]
        A = F.normalize(A, p=2, dim=1)
        B = F.normalize(B, p=2, dim=1)
        positive_score = torch.sum(A * B, dim=1)
        total_score = torch.matmul(A, B.permute(1, 0))
        positive_score = torch.exp(positive_score / self.c_temp)
        total_score = torch.sum(torch.exp(total_score / self.c_temp), axis=1)
        c_loss = - torch.mean(torch.log(positive_score / total_score))
        return c_loss

    def get_loss(self, learners_feature, courses_feature):
        concept_grained_learners_feature, course_grained_learners_feature = learners_feature
        concept_grained_course_feature, course_grained_courses_feature = courses_feature
        pred = torch.sum(concept_grained_learners_feature * concept_grained_course_feature, 2) + torch.sum(
            course_grained_learners_feature * course_grained_courses_feature, 2)
        bpr_loss = get_bpr_loss(pred)

        learner_contrastive_loss = self.get_contrastive_loss(concept_grained_learners_feature,
                                                             course_grained_learners_feature)
        course_contrastive_loss = self.get_contrastive_loss(concept_grained_course_feature,
                                                            course_grained_courses_feature)
        c_losses = [learner_contrastive_loss, course_contrastive_loss]
        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss

    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.get_concept_grained_graph()
            self.get_course_grained_graph()
            self.get_course_concept_graph()
        learners, courses = batch
        learners_feature, courses_feature = self.propagate()
        learners_embedding = [i[learners].expand(-1, courses.shape[1], -1) for i in learners_feature]
        courses_embedding = [i[courses] for i in courses_feature]
        bpr_loss, c_loss = self.get_loss(learners_embedding, courses_embedding)
        return bpr_loss, c_loss

    def evaluate(self, propagate_result, learners):
        learners_feature, courses_feature = propagate_result
        concept_grained_learners_feature, course_grained_learners_feature = [i[learners] for i in learners_feature]
        concept_grained_courses_feature, course_grained_courses_feature = courses_feature
        scores = torch.mm(concept_grained_learners_feature, concept_grained_courses_feature.t()) + torch.mm(
            course_grained_learners_feature,
            course_grained_courses_feature.t())
        return scores
