from models.recommender.NGCF.model import NGCFTrainer
from models.recommender.LightGCN.model import LightGCNTrainer
from models.recommender.NCF.model import NCFTrainer
from models.recommender import ItemCF, ItemAE, CML, UserAE, WMF
from functools import partial
import copy

shared_params = {
    #"device": 3,
}
""" RS model hyper-parameters."""
# dataset, target_item, device, path_fake_matrix
ngcf_args = {
    **shared_params,
    "trainer": NGCFTrainer,
    "embed_size": 64,
    "layer_size": '[64,64,64]',
    "mess_dropout": '[0.1,0.1,0.1]',
    "regs": '[1e-5]',
    "lr": 0.001,
    "epoch": 10,
    "batch_size": 1024,
}
ngcf_ml100k_args = copy.deepcopy(ngcf_args)
ngcf_ml100k_args['lr'] = 0.02
ngcf_ml100k_args['epoch'] = 30

ngcf_filmtrust_args = copy.deepcopy(ngcf_args)

ngcf_automotive_args = copy.deepcopy(ngcf_args)
ngcf_automotive_args['lr'] = 0.02
ngcf_automotive_args['epoch'] = 45

ngcf_ToolHome_args = copy.deepcopy(ngcf_args)
ngcf_ToolHome_args['lr'] = 0.02
ngcf_ToolHome_args['epoch'] = 40

ngcf_GroceryFood_args = copy.deepcopy(ngcf_args)
ngcf_GroceryFood_args['lr'] = 0.02
ngcf_GroceryFood_args['epoch'] = 30


ngcf_AppAndroid_args = copy.deepcopy(ngcf_args)

ngcf_yelp_args = copy.deepcopy(ngcf_args)
ngcf_yelp_args['lr'] = 0.05
ngcf_yelp_args['epoch'] = 10

lightgcn_args = {
    **shared_params,
    "trainer": LightGCNTrainer,
    "hidden_dim": 64,
    "n_layers": 1,
    "decay": 0.98,
    "reg": 0.001,
    "lr": 0.05,
    "minlr": 0.0001,
    "epochs": 45,
    "batch_size": 1024,
}
lightgcn_ml100k_args = copy.deepcopy(lightgcn_args)
lightgcn_ml100k_args['lr'] = 0.05
lightgcn_ml100k_args['epochs'] = 42

lightgcn_filmtrust_args = copy.deepcopy(lightgcn_args)

lightgcn_automotive_args = copy.deepcopy(lightgcn_args)
lightgcn_automotive_args['lr'] = 0.003
lightgcn_automotive_args['epochs'] = 42

lightgcn_ToolHome_args = copy.deepcopy(lightgcn_args)
lightgcn_ToolHome_args['lr'] = 0.002
lightgcn_ToolHome_args['epochs'] = 50

lightgcn_GroceryFood_args = copy.deepcopy(lightgcn_args)
lightgcn_GroceryFood_args['lr'] = 0.005
lightgcn_GroceryFood_args['epochs'] = 50

lightgcn_AppAndroid_args = copy.deepcopy(lightgcn_args)
lightgcn_yelp_args = copy.deepcopy(lightgcn_args)
lightgcn_yelp_args['lr'] = 0.005
lightgcn_yelp_args['epochs'] = 30


ncf_args = {
    **shared_params,
    "trainer": NCFTrainer,
    "n_factor": 32,
    "num_layers": 3,
    "num_ng": 4,
    "dropout": 0.0,
    "lr": 0.0005,
    "epochs": 10,
    "batch_size": 1024,
    "model_type": "NeuMF-end"
}
ncf_ml100k_args = copy.deepcopy(ncf_args)
ncf_ml100k_args['lr'] = 0.01
ncf_ml100k_args['epochs'] = 25

ncf_filmtrust_args = copy.deepcopy(ncf_args)

ncf_automotive_args = copy.deepcopy(ncf_args)
ncf_automotive_args['epochs'] = 7

ncf_ToolHome_args = copy.deepcopy(ncf_args)

ncf_GroceryFood_args = copy.deepcopy(ncf_args)
ncf_GroceryFood_args['epochs'] = 35

ncf_AppAndroid_args = copy.deepcopy(ncf_args)

ncf_yelp_args = copy.deepcopy(ncf_args)
ncf_yelp_args['lr'] = 0.02
ncf_yelp_args['epochs'] = 10


wmf_args = {
    **shared_params,
    "trainer": WMF.MFTrainer,
    "weight_alpha": None,
    "dim": 256,
    "lr": 0.02,  # 0.005
    "num_epoch": 50,
    "batch_size": 512,
}
wmf_ml100k_args = copy.deepcopy(wmf_args)
wmf_ml100k_args['lr'] = 0.01
wmf_ml100k_args['num_epoch'] = 40

wmf_filmtrust_args = copy.deepcopy(wmf_args)

wmf_automotive_args = copy.deepcopy(wmf_args)
wmf_automotive_args['lr'] = 0.005

wmf_ToolHome_args = copy.deepcopy(wmf_args)
wmf_ToolHome_args['lr'] = 0.2
wmf_ToolHome_args['num_epoch'] = 85

wmf_GroceryFood_args = copy.deepcopy(wmf_args)
wmf_GroceryFood_args['lr'] = 0.05
wmf_GroceryFood_args['num_epoch'] = 60

wmf_AppAndroid_args = copy.deepcopy(wmf_args)

wmf_yelp_args = copy.deepcopy(wmf_args)
wmf_yelp_args['lr'] = 0.2
wmf_yelp_args['num_epoch'] = 30


itemae_args = {
    **shared_params,
    "trainer": ItemAE.ItemAETrainer,
    "hidden_dims": '[256, 128]',
    "lr": 0.0001,  # 0.0001,
    "num_epoch": 10,  # 50,
    "batch_size": 128,
}
itemae_ml100k_args = copy.deepcopy(itemae_args)
itemae_filmtrust_args = copy.deepcopy(itemae_args)

itemae_automotive_args = copy.deepcopy(itemae_args)
itemae_automotive_args['lr'] = 0.0005
itemae_automotive_args['num_epoch'] = 8

itemae_ToolHome_args = copy.deepcopy(itemae_args)
itemae_ToolHome_args['lr'] = 0.01
itemae_ToolHome_args['num_epoch'] = 5

itemae_GroceryFood_args = copy.deepcopy(itemae_args)
itemae_GroceryFood_args['lr'] = 0.0005
itemae_GroceryFood_args['num_epoch'] = 15

itemae_AppAndroid_args = copy.deepcopy(itemae_args)

itemae_yelp_args = copy.deepcopy(itemae_args)
itemae_yelp_args['lr'] = 0.01
itemae_yelp_args['num_epoch'] = 15


itemcf_args = {
    **shared_params,
    "trainer": ItemCF.ItemCFTrainer,
    "num_epoch": 1,
    "knn": 50,
}
itemcf_ml100k_args = copy.deepcopy(itemcf_args)
itemcf_filmtrust_args = copy.deepcopy(itemcf_args)
itemcf_automotive_args = copy.deepcopy(itemcf_args)
itemcf_ToolHome_args = copy.deepcopy(itemcf_args)
itemcf_GroceryFood_args = copy.deepcopy(itemcf_args)
itemcf_AppAndroid_args = copy.deepcopy(itemcf_args)
itemcf_yelp_args = copy.deepcopy(itemcf_args)


cml_args = {
    **shared_params,
    "trainer": partial(CML.NCFTrainer, model_type='CML'),
    "hidden_dims": '[128]',
    "num_factors": 256,
    "lr": 0.01,
    "num_epoch": 20,
    "batch_size": 128,
}
cml_ml100k_args = copy.deepcopy(cml_args)
cml_filmtrust_args = copy.deepcopy(cml_args)

cml_automotive_args = copy.deepcopy(cml_args)
cml_automotive_args['lr'] = 0.05
cml_automotive_args['num_epoch'] = 5

cml_ToolHome_args = copy.deepcopy(cml_args)
cml_GroceryFood_args = copy.deepcopy(cml_args)
cml_AppAndroid_args = copy.deepcopy(cml_args)
cml_yelp_args = copy.deepcopy(cml_args)

# UserAETrainer, partial(UserAETrainer, model_type='CDAE')
userae_args = {
    **shared_params,
    "trainer": UserAE.UserAETrainer,
    "hidden_dims": '[600, 300]',
    "lr": 1e-4,
    "num_epoch": 20,
    "batch_size": 128,
}
userae_ml100k_args = copy.deepcopy(userae_args)
userae_ml100k_args['lr'] = 0.001
userae_ml100k_args['num_epoch'] = 45

userae_filmtrust_args = copy.deepcopy(userae_args)

userae_automotive_args = copy.deepcopy(userae_args)
userae_automotive_args['lr'] = 0.005
userae_automotive_args['num_epoch'] = 50

userae_ToolHome_args = copy.deepcopy(userae_args)
userae_ToolHome_args['lr'] = 0.0001
userae_ToolHome_args['num_epoch'] = 50

userae_GroceryFood_args = copy.deepcopy(userae_args)
userae_GroceryFood_args['lr'] = 0.001
userae_GroceryFood_args['num_epoch'] = 48


userae_AppAndroid_args = copy.deepcopy(userae_args)

userae_yelp_args = copy.deepcopy(userae_args)
userae_yelp_args['lr'] = 0.001
userae_yelp_args['num_epoch'] = 40

cdae_args = {
    **shared_params,
    "trainer": partial(UserAE.UserAETrainer, model_type='CDAE'),
    "hidden_dims": '[600, 300]',
    "lr": 0.001,
    "num_epoch": 45,
    "batch_size": 128,
}
cdae_ml100k_args = copy.deepcopy(cdae_args)
cdae_ml100k_args['lr'] = 0.005
cdae_ml100k_args['num_epoch'] = 15

cdae_filmtrust_args = copy.deepcopy(cdae_args)

cdae_automotive_args = copy.deepcopy(cdae_args)
cdae_automotive_args['lr'] = 0.001
cdae_automotive_args['num_epoch'] = 60

cdae_ToolHome_args = copy.deepcopy(cdae_args)
cdae_ToolHome_args['lr'] = 0.005
cdae_ToolHome_args['num_epoch'] = 15

cdae_GroceryFood_args = copy.deepcopy(cdae_args)
cdae_GroceryFood_args['lr'] = 0.005
cdae_GroceryFood_args['num_epoch'] = 10

cdae_AppAndroid_args = copy.deepcopy(cdae_args)
cdae_yelp_args = copy.deepcopy(cdae_args)
cdae_yelp_args['lr'] = 0.005
cdae_yelp_args['num_epoch'] = 50
