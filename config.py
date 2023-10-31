from pathlib import Path

def getConfig():
    return {
        "batch_size": 16,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 400,
        "d_model": 512,
        "lang_src": "de",
        "lang_tgt": "en",
        "model_folder": "models/",
        "model_fname": "model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/model",
    }


def get_weights_file_path(config, epoch):
    return str( 
        Path('.')
        + config["model_folder"]
        + config["model_fname"]
        + str(epoch)
        + "_of_"
        + str(config["num_epochs"])
        + ".pt"
    )
