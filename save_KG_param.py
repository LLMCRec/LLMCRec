from openke.config import Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader

dim = 128

train_dataloader = TrainDataLoader(
    in_path="./benchmarks/MOOCCubeX/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0,
    batch_size=16
)

distmult = DistMult(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim=dim
)

model = NegativeSampling(
    model=distmult,
    loss=SoftplusLoss(),
    batch_size=train_dataloader.get_batch_size(),
    regul_rate=1.0
)

distmult.load_checkpoint('./checkpoint/distmult-{}.ckpt'.format(dim))
tester = Tester(model=distmult, data_loader=train_dataloader, use_gpu=True)

distmult.save_parameters("./result/distmult-{}.json".format(dim), "./result/ent-dismult-{}.csv".format(dim))
