from models.attacker.attacker import Attacker

""" attacker model hyper-parameters."""
# dataset, target_item, device, path_atk_emb
atk_args = {
    "trainer": Attacker,
    "vic_rec": 'wmf',
    "n_factor": 64,
    "lr": 0.5,
    "num_epochs": 64,
    "batch_size": 64,
    "num_fake_user": 50,
    "num_filler_item": 50,
    "num_filler_pop": 30,
}
