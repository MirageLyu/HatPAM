from transformers import TransfoXLModel, TransfoXLTokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F


MODEL_PATH = "pretrained_models/"



def load_transformer_xl_tokenizer():
    tokenizer = TransfoXLTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    return tokenizer


def load_transformer_xl_model():
    model = TransfoXLModel.from_pretrained(MODEL_PATH, local_files_only=True)
    return model


# extra preprocessing methods
def text_preprocessing(text, tokenizer: TransfoXLTokenizer = None):
    # TODO: Add preprocess code here
    return text


def preprocess_for_transfoxl(text_arr, tokenizer: TransfoXLTokenizer = None):
    """
    :param text_arr: List of natural language text to be processed
    :param tokenizer: pre-trained tokenizer
    :return: embed sequence
    """
    input_ids = []

    if tokenizer is None:
        tokenizer = load_transformer_xl_tokenizer()

    tokenizer.pad_token = tokenizer.eos_token

    for i, sentence in enumerate(text_arr):
        encoded_sentence = tokenizer.encode_plus(
            text=text_preprocessing(sentence),
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=False
        )
        ids = encoded_sentence.get('input_ids')
        input_ids.append(ids)
        # atm = encoded_sentence.get('attention_mask')
        # attention_masks.append(atm)

    return input_ids # , torch.tensor(attention_mask)


class NaturalLanguageEmbeddingLayer(nn.Module):
    # freeze_bert: bool, set "False" to fine-tune the transfoxl model
    def __init__(self, freeze_bert=False, transfoxl_model=None, out_dimension=300):
        super(NaturalLanguageEmbeddingLayer, self).__init__()
        if transfoxl_model is None:
            self.transfoXL_model = load_transformer_xl_model()
        else:
            self.transfoXL_model = transfoxl_model
        if freeze_bert:
            for param in self.transfoXL_model.parameters():
                param.requires_pred = False

        # TODO: Is this necessary?
        self.forward_layer = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU(),
            nn.Linear(500, out_dimension)
        )

    def forward(self, inputs_ids):
        # print(inputs_ids.shape)
        inputs_ids = inputs_ids.view(1, inputs_ids.shape[-1])
        # print(inputs_ids.shape)
        x = self.transfoXL_model(inputs_ids)
        last_hidden_state_cls = x[0][:, 0, :]
        logits = self.forward_layer(last_hidden_state_cls)
        return logits


def initialize_natural_language_models():
    """
    :return: a transformer_xl model and a transformer_xl tokenizer
    """
    return load_transformer_xl_model(), load_transformer_xl_tokenizer()


# Check validation of transformer_xl model
if __name__ == '__main__':
    line = [open("dev_trans.txt", "r").readline(), open("dev_trans.txt", "r").readline()]
    transfoxl_model = load_transformer_xl_model()
    transfoxl_tokenizer = load_transformer_xl_tokenizer()
    input_ids = preprocess_for_transfoxl(line, tokenizer=transfoxl_tokenizer)
    nlel = NaturalLanguageEmbeddingLayer(freeze_bert=False, transfoxl_model=transfoxl_model)

    # print(line)
    # print(input_ids.shape)
    out = nlel(input_ids)
    # print(out.shape)
    # print(out[0][:, 0, :].shape)




