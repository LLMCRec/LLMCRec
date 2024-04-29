import torch
import argparse
import yaml
import os
import json
import torch.optim as optim
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from utility import Datasets
from models.LLMCRec import LLMCRec


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0", type=str, help="The GPU to use.")
    parser.add_argument("-d", "--dataset", default="MOOCCubeX-CS", type=str, help="The dataset to use.")
    parser.add_argument("-m", "--model", default="LLMCRec", type=str, help="The model to use.")
    parser.add_argument("-i", "--info", default="", type=str,
                        help="The additional information that will be shown in the log file name.")
    args = parser.parse_args()
    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("Config file loaded.")
    paras = parse_arguments().__dict__
    dataset_name = paras["dataset"]
    conf = conf[dataset_name]

    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    dataset = Datasets(conf)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    conf["num_learners"] = dataset.num_learners
    conf["num_courses"] = dataset.num_courses
    conf["num_concepts"] = dataset.num_concepts

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)

    for lr, l2_reg, course_grained_ratio, concept_grained_ratio, course_concept_ratio, embedding_size, layer_num, c_lambda, c_temp in \
            product(conf['lrs'], conf['l2_regs'], conf['course_grained_ratios'], conf['concept_grained_ratios'],
                    conf['course_concept_ratios'], conf["embedding_sizes"], conf["layer_nums"], conf["c_lambdas"],
                    conf["c_temps"]):
        log_path = "./logs/%s/%s" % (
            conf["dataset"], conf["model"])
        run_path = "./tensorboards/%s/%s" % (
            conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["dropout_interval"])]
        if conf["aug_type"] == "OP":
            assert course_grained_ratio == 0 and concept_grained_ratio == 0 and course_concept_ratio == 0

        settings += ["Neg_%d" % (conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg),
                     str(embedding_size)]

        conf["course_grained_ratio"] = course_grained_ratio
        conf["concept_grained_ratio"] = concept_grained_ratio
        conf["course_concept_ratio"] = course_concept_ratio
        conf["layer_num"] = layer_num
        settings += [str(course_grained_ratio), str(concept_grained_ratio), str(course_concept_ratio), str(layer_num)]

        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        settings += [str(c_lambda), str(c_temp)]

        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        run = SummaryWriter(run_path)

        if conf['model'] == 'LLMCRec':
            model = LLMCRec(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" % (conf["model"]))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        batch_count = len(dataset.train_loader)
        test_interval_bs = int(batch_count * conf["test_interval"])
        dropout_interval_bs = int(batch_count * conf["dropout_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf['epochs']):
            epoch_anchor = epoch * batch_count
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor + 1) % dropout_interval_bs == 0:
                    ED_drop = True
                bpr_loss, c_loss = model(batch, ED_drop=ED_drop)
                loss = bpr_loss + conf["c_lambda"] * c_loss
                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" % (
                    epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {"val": test(model, dataset.val_loader, conf),
                               "test": test(model, dataset.test_loader, conf)}
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path,
                                                                         checkpoint_model_path, checkpoint_conf_path,
                                                                         epoch, batch_anchor, best_metrics,
                                                                         best_perform, best_epoch)


def init_best_metrics(conf):
    best_metrics = {"val": {}, "test": {}}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for top_k in conf['top_k']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][top_k] = 0
    best_perform = {"val": {}, "test": {}}
    return best_metrics, best_perform


def write_log(run, log_path, top_k, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" % (m, top_k), val_score[top_k], step)
        run.add_scalar("%s_%d/Test" % (m, top_k), test_score[top_k], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (
        curr_time, top_k, val_scores["recall"][top_k], val_scores["ndcg"][top_k])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (
        curr_time, top_k, test_scores["recall"][top_k], test_scores["ndcg"][top_k])

    log = open(log_path, "a")
    log.write("%s\n" % val_str)
    log.write("%s\n" % test_str)
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor,
                best_metrics, best_perform, best_epoch):
    for top_k in conf["top_k"]:
        write_log(run, log_path, top_k, batch_anchor, metrics)

    log = open(log_path, "a")

    top_k_judge = 20
    print("top%d as the final evaluation standard" % top_k_judge)
    if (metrics["val"]["recall"][top_k_judge] > best_metrics["val"]["recall"][top_k_judge] and
            metrics["val"]["ndcg"][top_k_judge] > best_metrics["val"]["ndcg"][top_k_judge]):
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for top_k in conf['top_k']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][top_k] = metrics[key][metric][top_k]

            best_perform["test"][top_k] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" % (
                curr_time, best_epoch, top_k, best_metrics["test"]["recall"][top_k],
                best_metrics["test"]["ndcg"][top_k])
            best_perform["val"][top_k] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" % (
                curr_time, best_epoch, top_k, best_metrics["val"]["recall"][top_k], best_metrics["val"]["ndcg"][top_k])
            print(best_perform["val"][top_k])
            print(best_perform["test"][top_k])
            log.write(best_perform["val"][top_k] + "\n")
            log.write(best_perform["test"][top_k] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for top_k in conf["top_k"]:
            tmp_metrics[m][top_k] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for learners, learner_course_ground_truth, learner_course_mask in dataloader:
        pred_b = model.evaluate(rs, learners.to(device))
        pred_b -= 1e8 * learner_course_mask.to(device)
        tmp_metrics = get_metrics(tmp_metrics, learner_course_ground_truth.to(device), pred_b, conf["top_k"])

    metrics = {}
    for m, top_k_res in tmp_metrics.items():
        metrics[m] = {}
        for top_k, res in top_k_res.items():
            metrics[m][top_k] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, top_ks):
    tmp = {"recall": {}, "ndcg": {}}
    for top_k in top_ks:
        _, col_indices = torch.topk(pred, top_k)
        row_indices = torch.zeros_like(col_indices) + torch.arange(pred.shape[0], device=pred.device,
                                                                   dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indices.view(-1), col_indices.view(-1)].view(-1, top_k)

        tmp["recall"][top_k] = get_recall(pred, grd, is_hit)
        tmp["ndcg"][top_k] = get_ndcg(pred, grd, is_hit, top_k)

    for m, top_k_res in tmp.items():
        for top_k, res in top_k_res.items():
            for i, x in enumerate(res):
                metrics[m][top_k][i] += x

    return metrics


def get_recall(pred, grd, is_hit):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, top_k):
    def get_dcg(hit, top_k, device):
        hit = hit / torch.log2(torch.arange(2, top_k + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def get_idcg(num_pos, top_k, device):
        hit = torch.zeros(top_k, dtype=torch.float).to(device)
        hit[:num_pos] = 1
        return get_dcg(hit, top_k, device)

    device = grd.device
    idcgs = torch.empty(1 + top_k, dtype=torch.float)
    idcgs[0] = 1
    for i in range(1, top_k + 1):
        idcgs[i] = get_idcg(i, top_k, device)

    num_pos = grd.sum(dim=1).clamp(0, top_k).to(torch.long)
    dcg = get_dcg(is_hit, top_k, device)

    idcg = idcgs.to(device)[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
