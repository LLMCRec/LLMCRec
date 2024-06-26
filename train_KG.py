from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

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

test_dataloader = TestDataLoader("./benchmarks/MOOCCubeX/", "link")

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

trainer = Trainer(model=model, data_loader=train_dataloader, train_times=2000, alpha=0.5, use_gpu=True,
                  opt_method="adagrad")
trainer.run()
distmult.save_checkpoint('./checkpoint/distmult-{}.ckpt'.format(dim))

distmult.load_checkpoint('./checkpoint/distmult-{}.ckpt'.format(dim))
tester = Tester(model=distmult, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)
