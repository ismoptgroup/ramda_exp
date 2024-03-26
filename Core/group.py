# group

import torch

def group_model(model: torch.nn.Module, name: str, lambda_: float):
    if name == "Lin":
        optimizer_grouped_parameters = [
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        }]  
        
        for n, p in model.named_parameters():
            if p.ndim == 2:
                optimizer_grouped_parameters[0]["names"].append(n)
                optimizer_grouped_parameters[0]["params"].append(p)
            else:
                optimizer_grouped_parameters[1]["names"].append(n)
                optimizer_grouped_parameters[1]["params"].append(p)
                
    elif name == "VGG" or name == "ResNet" or name == "LeNet":
        optimizer_grouped_parameters = [
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0,2,3),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        }]

        for n, p in model.named_parameters():
            if p.ndim == 4:
                optimizer_grouped_parameters[0]["names"].append(n)
                optimizer_grouped_parameters[0]["params"].append(p)
            elif p.ndim == 2:
                optimizer_grouped_parameters[1]["names"].append(n)
                optimizer_grouped_parameters[1]["params"].append(p)
            else:
                optimizer_grouped_parameters[2]["names"].append(n)
                optimizer_grouped_parameters[2]["params"].append(p)

    elif name == "Transformer-XL":
        Linear = ["dec_attn.qkv_net.weight", "dec_attn.o_net.weight", "dec_attn.r_net.weight", "pos_ff.CoreNet.0.weight", "pos_ff.CoreNet.3.weight"]
        optimizer_grouped_parameters = [
            {
                "names": [n for n, p in model.named_parameters() if any(nd in n for nd in Linear)],
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in Linear)],
                "lambda_": lambda_,
                "dim": (0),
                "dim_": (1)
            },
            {
                "names": [n for n, p in model.named_parameters() if not any(nd in n for nd in Linear)],
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in Linear)],
                "lambda_": 0.0,
                "dim": None,
                "dim_": None
            },
        ]

    elif name == "Tacotron2":
        Conv1d = ["conv.weight"]
        Linear = ["linear_layer.weight"]
        LSTM = ["lstm.weight", "rnn.weight"]
        optimizer_grouped_parameters = [
        {
            "names": [n for n, p in model.named_parameters() if any(nd in n for nd in Conv1d)],
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in Conv1d)],
            "lambda_": lambda_,
            "dim": (0,2),
            "dim_": (1)
        },
        {
            "names": [n for n, p in model.named_parameters() if any(nd in n for nd in Linear)],
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in Linear)],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [n for n, p in model.named_parameters() if any(nd in n for nd in LSTM)],
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in LSTM)],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [n for n, p in model.named_parameters() if not any(nd in n for nd in Conv1d) and not any(nd in n for nd in Linear) and not any(nd in n for nd in LSTM)],
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in Conv1d) and not any(nd in n for nd in Linear) and not any(nd in n for nd in LSTM)],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        },
        ]
    elif name == "Bert":
        optimizer_grouped_parameters = [
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        }]

        for n, p in model.named_parameters():
            if p.ndim == 2 and n != "bert.embeddings.word_embeddings.weight" and n != "bert.embeddings.position_embeddings.weight" and n != "bert.embeddings.token_type_embeddings.weight":
                optimizer_grouped_parameters[0]["names"].append(n)
                optimizer_grouped_parameters[0]["params"].append(p)
            else:
                optimizer_grouped_parameters[1]["names"].append(n)
                optimizer_grouped_parameters[1]["params"].append(p)

    elif name == "ViT":
        optimizer_grouped_parameters = [
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        }]

        for n, p in model.named_parameters():
            if p.ndim == 2:
                optimizer_grouped_parameters[0]["names"].append(n)
                optimizer_grouped_parameters[0]["params"].append(p)
            else:
                optimizer_grouped_parameters[1]["names"].append(n)
                optimizer_grouped_parameters[1]["params"].append(p) 
    elif name == "MLP":
        optimizer_grouped_parameters = [
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        }]

        for n, p in model.named_parameters():
            if p.ndim == 2:
                optimizer_grouped_parameters[0]["names"].append(n)
                optimizer_grouped_parameters[0]["params"].append(p)
            else:
                optimizer_grouped_parameters[1]["names"].append(n)
                optimizer_grouped_parameters[1]["params"].append(p)

    return optimizer_grouped_parameters
