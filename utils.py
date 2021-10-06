import json
import os
import unicodedata

from functools import reduce
from collections import deque
import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass, asdict, is_dataclass
from itertools import chain
from typing import Dict, List, Set, Tuple, Union
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import random

@dataclass(eq=True, frozen=True)
class Evidence:
    """
    (docid, start_character, end_character) form the only official Evidence; sentence level annotations are for convenience.
    Args:
        text: Some representation of the evidence text
        docid: Some identifier for the document
        start_character: The canonical start character, inclusive
        end_character: The canonical end token, exclusive
        start_sentence: Best guess start sentence, inclusive
        end_sentence: Best guess end sentence, exclusive
    """
    text: Union[str, List[int]]
    docid: str
    start_char: int=-1
    end_char: int=-1
    start_sentence: int=-1
    end_sentence: int=-1


@dataclass(eq=True, frozen=True)
class Annotation:
    """
    Args:
        annotation_id: unique ID for this annotation element
        query: some representation of a query string
        evidences: a set of "evidence groups". 
            Each evidence group is:
                * sufficient to respond to the query (or justify an answer)
                * composed of one or more Evidences
                * may have multiple documents in it (depending on the dataset)
                    - e-snli has multiple documents
                    - other datasets do not
        classification: str
        query_type: Optional str, additional information about the query
        docids: a set of docids in which one may find evidence.
    """
    annotation_id: str
    query: Union[str, List[int]]
    evidences: Set[Tuple[Evidence]]
    classification: str
    query_type: str = None
    docids: Set[str] = None

    def all_evidences(self) -> Tuple[Evidence]:
        return tuple(list(chain.from_iterable(self.evidences)))
    

@dataclass(eq=True, frozen=True)
class Instance:
    """
    """
    token_embeddings: Tensor
    rationale: List[int]
    kept_tokens: List[int]
    label: str
    annotation_id: str


def annotations_to_jsonl(annotations, output_file, mode='w'):
    with open(output_file, mode, encoding='utf-8') as of:
        for ann in sorted(annotations, key=lambda x: x.annotation_id):
            as_json = _annotation_to_dict(ann)
            as_str = json.dumps(as_json, sort_keys=True, ensure_ascii=False).encode('utf8')
            of.write(as_str.decode())
            of.write('\n')

def _annotation_to_dict(dc):
    # convenience method
    if is_dataclass(dc):
        d = asdict(dc)
        ret = dict()
        for k, v in d.items():
            ret[k] = _annotation_to_dict(v)
        return ret
    elif isinstance(dc, dict):
        ret = dict()
        for k, v in dc.items():
            k = _annotation_to_dict(k)
            v = _annotation_to_dict(v)
            ret[k] = v
        return ret
    elif isinstance(dc, str):
        return dc
    elif isinstance(dc, (set, frozenset, list, tuple)):
        ret = []
        for x in dc:
            ret.append(_annotation_to_dict(x))
        return tuple(ret)
    else:
        return dc

def preprocess_line(line: str) -> str:
    line = unicodedata.normalize("NFKD", line) # load french spaces properly
    line = line.replace('\xad', '')  # replace soft-hypens in multirc
    line = line.strip()
    return line

def load_jsonl(fp: str) -> List[dict]:
    ret = []
    with open(fp, 'r', encoding='utf-8') as inf:
        for line in inf:
            line = preprocess_line(line)
            # line = unicodedata.normalize("NFKD", line)  # load french spaces properly
            js = json.loads(line)
            ret.append(js)
    return ret

def write_jsonl(jsonl, output_file):
    with open(output_file, 'w', encoding='utf-8') as of:
        for js in jsonl:
            as_str = json.dumps(js, sort_keys=True, ensure_ascii=False)
            of.write(as_str)
            of.write('\n')

def annotations_from_jsonl(fp: str) -> List[Annotation]:
    ret = []
    with open(fp, 'r', encoding='utf-8') as inf:
        for line in inf:
            # line = unicodedata.normalize("NFKD", line)  # load french spaces properly
            # line = line.replace('\xad', '')  # replace soft-hypens in multirc
            line = preprocess_line(line)
            content = json.loads(line)
            ev_groups = []
            for ev_group in content['evidences']:
                ev_group = tuple([Evidence(**ev) for ev in ev_group]) 
                ev_groups.append(ev_group)
            content['evidences'] = frozenset(ev_groups)
            ret.append(Annotation(**content))
    return ret

def load_datasets(data_dir: str) -> Tuple[List[Annotation], List[Annotation], List[Annotation]]:
    """Loads a training, validation, and test dataset

    Each dataset is assumed to have been serialized by annotations_to_jsonl,
    that is it is a list of json-serialized Annotation instances.
    """
    train_data = annotations_from_jsonl(os.path.join(data_dir, 'train.jsonl'))
    val_data = annotations_from_jsonl(os.path.join(data_dir, 'val.jsonl'))
    test_data = annotations_from_jsonl(os.path.join(data_dir, 'test.jsonl'))
    return train_data, val_data, test_data

def load_documents(data_dir: str, docids: Set[str]=None) -> Dict[str, str]:
    """Loads a subset of available documents from disk. 
    """

    if os.path.exists(os.path.join(data_dir, 'docs.jsonl')) :
        assert not os.path.exists(os.path.join(data_dir, 'docs'))
        return load_documents_from_file(data_dir, docids)
        
    docs_dir = os.path.join(data_dir, 'docs')
    res = dict()
    if docids is None:
        docids = sorted(os.listdir(docs_dir))
    else:
        docids = sorted(set(str(d) for d in docids))
    for d in docids:
        # cannot simply use read() in order to follow read in same string as eraser; eraser does some preprocessing to the string
        with open(os.path.join(docs_dir, d), 'r', encoding='utf-8') as inf:  
            lines = [preprocess_line(l) for l in inf.readlines()]
            lines = list(filter(lambda x: bool(len(x)), lines))
            tokenized = [list(filter(lambda x: bool(len(x)), line.strip().split(' '))) for line in lines]
            flattened_tokenized = list(chain.from_iterable(tokenized))
            res[d] = " ".join(flattened_tokenized)

            # res[d] = unicodedata.normalize("NFKD", inf.read().splitlines())
    return res

def load_documents_from_file(data_dir: str, docids: Set[str]=None) -> Dict[str, str]:
    """Loads a subset of available documents from 'docs.jsonl' file on disk.

    Each document is assumed to be serialized as newline ('\n') separated sentences.
    Each sentence is assumed to be space (' ') joined tokens.
    """
    docs_file = os.path.join(data_dir, 'docs.jsonl')
    documents = load_jsonl(docs_file)
    documents = {doc['docid']: doc['document'] for doc in documents}
    res = dict()
    if docids is None:
        docids = sorted(list(documents.keys()))
    else:
        docids = sorted(set(str(d) for d in docids))
    for d in docids:
        res[d] = documents[d]
    return res

def read_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        js = json.load(f)
    return js

def write_json(js, fp):
    with open(fp, 'w+', encoding='utf-8') as f:
        json.dump(js, f, sort_keys=True, ensure_ascii=False, indent=4)

def generate_document_evidence_map(evidences: List[List[Evidence]]) -> Dict[str, Tuple[int, int]]:
    document_evidence_map = {}
    for evgroup in evidences:
        for evclause in evgroup:
            if evclause.docid not in document_evidence_map:
                document_evidence_map[evclause.docid] = []
            document_evidence_map[evclause.docid].append((evclause.start_char, evclause.end_char))

    return document_evidence_map

def generate_document_evidence_string_map(evidences: List[List[Evidence]]) -> Dict[str, List[str]]:
    """ For assertion purpose. """

    document_evidence_map = {}
    for evgroup in evidences:
        for evclause in evgroup:
            if evclause.docid not in document_evidence_map:
                document_evidence_map[evclause.docid] = []
            document_evidence_map[evclause.docid].append(evclause.text)

    return document_evidence_map

def is_subspan(subspan: Tuple[int], span: Tuple[int]) -> bool:
    assert len(subspan) == 2
    assert len(span) == 2
    return subspan[0] >= span[0] and subspan[1] <= span[1]

def get_model_device(model):
    return next(model.parameters()).device

def get_wordpiece_embeddings(inputs: Dict, model) -> Tensor:
    """ Based on https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#2-input-formatting """
    with torch.no_grad():
        inputs = {k: v.to(get_model_device(model)) for k, v in inputs.items()}
        outputs = model(**inputs)
        hidden_states = outputs[2]  # 13x1xtx768

    # stack list of tensors 
    wordpiece_embeddings = torch.stack(hidden_states[-4:], dim = 0)  # 4x1xtx768

    # remove batch dimension
    wordpiece_embeddings = torch.squeeze(wordpiece_embeddings, dim = 1)  # 4xtx768

    # order by wordpiece tokens
    wordpiece_embeddings = wordpiece_embeddings.permute(1, 0, 2)  # tx4x768

    # concat last four hidden layers as in original paper
    s = wordpiece_embeddings.size()
    wordpiece_embeddings = wordpiece_embeddings.reshape(s[0], s[1] * s[2])  # tx3072
    
    return wordpiece_embeddings

def merge_wordpiece_embeddings(embeddings: Tensor, word_ids: List) -> Tensor:
    """ Merges wordpiece embeddings by averaging them. """

    result = []
    stack = [0]  # initialise stack
    test_embedding = embeddings[0]  # for assertion of sizes

    for i in range(1, len(word_ids)):
        # if previous id is different, then average all tensors in accumulated tensors
        if word_ids[i] == None or word_ids[i] != word_ids[i-1]: # None refer to special tokens, which will never be merged
            # accumulate tensors
            avg_e = torch.mean(embeddings[stack[0]: stack[-1] + 1], 0)
            result.append(avg_e)
            assert avg_e.size() == test_embedding.size(), f"Sizes differ: {avg_e.size()} != {test_embedding.size()}"

            # initialise stack with current idx
            stack = [i]
            pass
        else: 
            stack.append(i)

    # clear remaining accumulated tensors
    avg_e = torch.mean(embeddings[stack[0]: stack[-1] + 1], 0)
    result.append(avg_e)
    assert avg_e.size() == test_embedding.size(), f"Sizes differ: {avg_e.size()} != {test_embedding.size()}"

    # stack result
    result = torch.stack(result, dim = 0)
    
    # sanity check that all wordpiece embeddings are merged (i.e. duplicated elements in word_ids, excluding None)
    assert len(set(word_ids)) + word_ids.count(None) - 1 == result.size()[0]  
    return result

def merge_character_spans(batch_encoding) -> List[Tuple[int]]:
    """ Merges character spans of wordpiece embeddings by expanding them. """
    word_ids = batch_encoding.word_ids()
    result = []
    stack = [0]  # initialise stack

    for i in range(1, len(word_ids)):
        # if previous id is different, then merge character spans of tokens in stack
        if word_ids[i] == None or word_ids[i] != word_ids[i-1]: # None refer to special tokens, which will never be merged
            if word_ids[i-1] == None: 
                assert len(stack) == 1
                result.append((-1, -1))  # special case as special tokens have no character spans
            else:
                # merge character spans
                s_i = batch_encoding.token_to_chars(stack[0])[0]
                e_i = batch_encoding.token_to_chars(stack[-1])[1]
                assert s_i < e_i
                result.append((s_i, e_i))

            # initialise stack with current idx
            stack = [i]
            pass
        else: 
            stack.append(i)

    # clear remaining. Should be left with last token, which is a special token
    assert len(stack) == 1 and word_ids[stack[0]] == None
    result.append((-1, -1))
    
    # sanity check that all wordpiece embeddings are merged (i.e. duplicated elements in word_ids, excluding None)
    assert len(set(word_ids)) + word_ids.count(None) - 1 == len(result) 
    return result

def create_instance(ann: Annotation, docs: Dict[str, str], tokenizer, embedding_model, logger=None):
    annotation_id: str = ann.annotation_id
    evidences: List[List[Evidence]] = ann.evidences
    label: str = str(ann.classification)
    query: str = ann.query 

    docids: List[str] = sorted([f"{annotation_id}_premise", f"{annotation_id}_hypothesis"]) # only for esnli
    # docids_2: List[str] = sorted(list(set([evclause.docid for evgroup in evidences for evclause in evgroup])))  # easily overfit esnli: contradiction both premise and hypothesis, neutral only hypothesis.
    # assert docids == docids_2, f"{docids} != {docids_2}"

    document_evidence_map: Dict[str, List[Tuple[int, int]]] = generate_document_evidence_map(evidences)
    document_evidence_str_map: Dict[str, List[str]] = generate_document_evidence_string_map(evidences)  # for assertion later
    assert set(document_evidence_map.keys()).issubset(set(docids)), "Evidence should come from docids!"

    token_embeddings = []
    rationale = []
    # TODO: kept_tokens doesn't make sense from backprop persp - deterministic alteration to op fn.
    # kept_tokens = []

    for docid in docids:
        # tokenizer - return special tokens mask information for always kept.
        inputs = tokenizer(docs[docid], truncation=True, return_tensors="pt", add_special_tokens = False)
        tokens = tokenizer.tokenize(docs[docid], truncation=True, add_special_tokens = False)  # for sanity check later
        tokens = [t.strip("#") for t in tokens]
        logger.debug(tokens)

        # get wordpiece embeddings
        wp_e = get_wordpiece_embeddings(inputs, embedding_model)
        token_embeddings.extend(wp_e)

        # generate rationale from wordpiece embeddings
        r = [0] * wp_e.size()[0]
        # r = torch.zeros(wp_e.size()[0])
        # special_tokens_mask = [int(id == None) for id in inputs.word_ids()]  # proxy for special tokens mask. None == Special Tokens. Integer = otherwise.
        # assert len(r) == len(special_tokens_mask)
        assert len(r) == len(tokens)
        if docid in document_evidence_map:
            for i in range(len(r)):
                # if special_tokens_mask[i]: r[i] = 0  # special token
                # else:  # wordpieces
                wp_span = inputs.token_to_chars(i)
                is_rat = reduce(lambda prev, span: prev or is_subspan(wp_span, span), document_evidence_map[docid], False)
                if is_rat:
                    r[i] = 1 
                    is_token_in_evd = reduce(lambda prev, evd: prev or tokens[i] in evd, document_evidence_str_map[docid], False)
                    if not is_token_in_evd and tokens[i] != '[UNK]':
                        logger.warning(f"{tokens[i]} is not found in evidence {document_evidence_str_map}! annotation_id: {annotation_id}")
                        logger.warning(tokens)
                        # raise ValueError
                    # if not is_token_in_evd: logger.warn(f"{tokens[i]} is not found in evidence {document_evidence_str_map}!")
                else:
                    r[i] = 0

        rationale.extend(r)

        # generate kept tokens
        # k_t = special_tokens_mask  
        # kept_tokens.extend(k_t)
    
    if query != "" and type(query) != list:
        assert False, f"Only e-snli supported for now.: {query}"
        inputs = tokenizer(query, return_tensors="pt")
        wp_e = get_wordpiece_embeddings(inputs, embedding_model)
        token_embeddings.extend(wp_e)
        rationale.extend([1] * wp_e.size()[0])  # query are always considered evidence
        kept_tokens.extend([1] * wp_e.size()[0])

    token_embeddings = torch.stack(token_embeddings, dim = 0)
    rationale = torch.tensor(rationale)
    assert token_embeddings.size()[0] == len(rationale)

    # i = Instance(token_embeddings, rationale, kept_tokens, label, annotation_id)
    # i = {
    #     "token_embeddings": token_embeddings,
    #     "rationale": rationale,
    #     "kept_tokens": kept_tokens,
    #     "label": label,
    #     "annotation_id": annotation_id
    # }

    return token_embeddings, rationale, label, annotation_id

def load_instances(data_dir, tokenizer, embedding_model, logger, debug=False):
    documents: Dict[str, str] = load_documents(data_dir, docids=None)
    splits = ["train", "val", "test"]
    ret = []
    for split in splits:
        logger.info(f"Loading {split}.jsonl...")
        annotations = annotations_from_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
        instances: List[Instance] = []

        for _, line in enumerate(tqdm(annotations)):
            i = create_instance(line, documents, tokenizer, embedding_model, logger=logger)
            if debug == True: i = 0  # dummy value. purpose is to test if create_instance breaks for whole training set.
            logger.debug(i)
            instances.append(i)

        ret.append(instances)

        # # caching
        # torch.save(instances, os.path.join(args.data_dir, f"cached_{split}_{args.model_name}_instances"))
    if debug == True: 
        print("Success!")
        exit(1)
    return ret

def get_num_classes(dataset_name):
    return len(dataset_mapping[dataset_name])

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_optimizer(models, lr=1e-3):
    '''
        -models: List of models (such as Generator, classif, memory, etc)
        -args: experiment level config

        returns: torch optimizer over models
    '''
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    return torch.optim.Adam(params, lr=lr)

def get_base_dataset_name(dataset_name):
    if "esnli" in dataset_name: return "esnli"
    elif "multirc" in dataset_name: return "multirc"
    else: raise NotImplementedError

dataset_mapping = {
    "multirc": {
        "False": 0,
        "True": 1
    },
    "esnli": {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2
    }
}

def tracked_named_parameters(named_parameters):
    res = [(n, p) for n, p in named_parameters if(p.requires_grad) and ("bias" not in n)]
    return res

def plot_grad_flow(mean_grads, var_grads, layers, fp):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.'''
    # ave_grads = []
    # max_grads= []
    # layers = []
    # for n, p in named_parameters:
    #     if(p.requires_grad) and ("bias" not in n):
    #         layers.append(n)
    #         ave_grads.append(p.grad.abs().mean())
    #         max_grads.append(p.grad.abs().max())

    assert len(mean_grads) == len(var_grads) == len(layers)
    plt.clf()
    plt.bar(np.arange(len(mean_grads)), mean_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(mean_grads)), var_grads, alpha=0.1, lw=1, color="b")
    plt.xticks(range(0,len(mean_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(mean_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.savefig(fp, bbox_inches='tight')

def score_hard_rationale_predictions(mask_pad, rationale_pad, token_lengths):
    """ Follows ERASER paper's implementation. """
    assert mask_pad.size() == rationale_pad.size()  # (L, N) == (L, N)
    running_scores = torch.zeros(3)
    for i in range(len(token_lengths)):
        mask_instance = mask_pad[:token_lengths[i], i]
        rationale_instance = rationale_pad[:token_lengths[i], i]
        # print(f"mask instance: {mask_instance}")
        # print(f"rat instance: {rationale_instance}")
        assert len(mask_instance) == len(rationale_instance)
        scores = precision_recall_fscore_support(rationale_instance, mask_instance, average='binary', zero_division=0)
        running_scores += torch.tensor(scores[:-1])

    p, r, f1 = running_scores / len(token_lengths)
    return p, r, f1


### TESTS

def test_get_wordpiece_embeddings():
    from transformers import AutoModel
    from transformers import AutoTokenizer
    """ Tests for contextual embeddings. Follows https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#2-input-formatting. """

    from scipy.spatial.distance import cosine

    model_name = 'bert-base-uncased'
    # model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
    model.eval()
    
    # english
    en = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    inputs = tokenizer(en, return_tensors="pt")
    we_en = get_wordpiece_embeddings(inputs, model)

    tokens = tokenizer.tokenize(en, add_special_tokens = True)
    ids = [i for i, x in enumerate(tokens) if x == "bank"]

    same_bank = 1 - cosine(we_en[ids[0]], we_en[ids[1]])
    diff_bank = 1 - cosine(we_en[ids[0]], we_en[ids[2]])
    print(f"diff_bank vs same_bank for en: {diff_bank} vs {same_bank}")
    
    assert abs(same_bank - 0.93890) < 0.0001 
    assert abs(diff_bank - 0.69093) < 0.0001 

def test_merge_wordpiece_embeddings():
    embeddings = torch.rand(26, 13, 768)
    word_ids = [None, 0, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, None]
    merged_embeddings = merge_wordpiece_embeddings(embeddings, word_ids)
    assert merged_embeddings.size()[0] == len(set(word_ids)) + 1  # +1 to account for None

    embeddings = torch.tensor([[0., 0.], [1., -1.], [2., -2.], [3., -3.]])  # 4x2
    word_ids = [None, 0, 0, None]
    merged_embeddings = merge_wordpiece_embeddings(embeddings, word_ids)
    expected_embeddings = torch.tensor([[0., 0.], [1.5, -1.5], [3, -3]])
    assert torch.equal(merged_embeddings, expected_embeddings)

def test_merge_character_spans():
    from transformers import AutoTokenizer
    import nltk
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fr = "Après avoir volé de l'argent dans le coffre-fort de la banque, le braqueur de banque a été vu en train de pêcher sur les rives du Mississippi."
    inputs = tokenizer(fr, return_tensors="pt")
    character_spans = merge_character_spans(inputs)
    character_spans = [span for span in character_spans if span != (-1, -1)]  # filter out special tokens

    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    token_spans = list(tokenizer.span_tokenize(fr))
    print(character_spans)
    print(token_spans)
    assert character_spans == token_spans

def test_gen_loss():
    from models.generator import Generator
    config = {
        "lstm": {
            "input_size": 3072,
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True
        },
        "dropout": 0.2,
        "linear": {
            "in_features": 256
        },
        "selection_lambda": 1,
        "continuity_lambda": 1
    }
    gen = Generator(config)

    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 0, 1, 1]], dtype=torch.float).T  # 5x2 ~= (L, N)
    selection_cost, continuity_cost = gen.loss(mask)
    assert selection_cost == 3, selection_cost
    assert continuity_cost == 2.5, continuity_cost

def test_score_hard_rationale_predictions():
    mask = torch.tensor([[0, 1, 1, 1, 0], [1, 0, 1, 1, 0]]).T
    rationale = torch.tensor([[1, 0, 1, 0, 0], [1, 1, 0, 0, 0]]).T
    token_lengths = [4, 3]
    p, r, f1 = score_hard_rationale_predictions(mask, rationale, token_lengths)
    assert abs(p.item() - 0.4167) <= 0.0001, p
    assert abs(r.item() - 0.5) == 0, r

if __name__ == "__main__":
    print("Running unit tests...")
    # test_merge_character_spans()
    # test_merge_wordpiece_embeddings()
    test_gen_loss()
    test_score_hard_rationale_predictions()
    test_get_wordpiece_embeddings()
    print("Unit tests passed!")


