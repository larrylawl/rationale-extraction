import json
import os
import unicodedata
import nltk

from functools import reduce
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
from models.encoder import Encoder
from models.generator import Generator

@dataclass(eq=True, frozen=True)
class Evidence:
    """
    (docid, start_char, end_char) form the only official Evidence; sentence level annotations are for convenience.
    Args:
        text: Some representation of the evidence text
        docid: Some identifier for the document
        start_char: The canonical start char, inclusive
        end_char: The canonical end token, exclusive
        start_sentence: Best guess start sentence, inclusive
        end_sentence: Best guess end sentence, exclusive
    """
    text: Union[str, List[int]]
    docid: str
    # start_char: int=-1
    # end_char: int=-1
    start_token: int = -1
    end_token: int = -1
    start_sentence: int=-1
    end_sentence: int=-1


@dataclass(eq=True)
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
    alignment: Dict[int, List[int]] = None

    def all_evidences(self) -> Tuple[Evidence]:
        return tuple(list(chain.from_iterable(self.evidences)))

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
            js = json.loads(line)
            ret.append(js)
    return ret

def load_id_jsonl_as_dict(fp: str) -> Dict:
    """ Loads jsonl with doc id as a dictionary that maps docid to dictionary that contains the      remaining key-value pairs of the original json.
    """
    op = {}
    with open(fp, 'r', encoding='utf-8') as inf:
        for line in inf:
            line = preprocess_line(line)
            js = json.loads(line)
            docid = js["docid"]
            del js["docid"]
            assert docid not in op, "Document ids should be unique!"
            op[docid] = js
    return op

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

def load_flattened_documents(data_dir: str, docids: Set[str]=None) -> Dict[str, List[str]]:
    """Loads a subset of available documents from disk.

    Returns a tokenized version of the document.
    """
    unflattened_docs = load_documents(data_dir, docids)
    flattened_docs = dict()
    for doc, unflattened in unflattened_docs.items():
        flattened_docs[doc] = list(chain.from_iterable(unflattened))
    return flattened_docs

def load_documents_from_file(data_dir: str, docids: Set[str]=None) -> Dict[str, str]:
    """Loads a subset of available documents from 'docs.jsonl' file on disk.
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
            # document_evidence_map[evclause.docid].append((evclause.start_char, evclause.end_char))
            document_evidence_map[evclause.docid].append((evclause.start_token, evclause.end_token))

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

def get_token_embeddings(string, tokenizer, embedding_model, merge_strategy = "average"):
    """ Retrieves token embeddings by accumulating wordpiece embeddings based on merge strategy.
    Identify wordpiece to tokens by checking if their character span is in subset of the original token char span. """
    def _merge_embeddings(wp_e, stack):
        if merge_strategy == "average": 
            t_e = torch.mean(wp_e[stack], 0)
        elif merge_strategy == "first":
            t_e = wp_e[stack[0]]
        else:
            raise NotImplementedError
        return t_e


    inputs = tokenizer(string, truncation=True, return_tensors="pt", add_special_tokens = False)
    wp_e = get_wordpiece_embeddings(inputs, embedding_model)

    ws_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    token_spans = ws_tokenizer.span_tokenize(string)

    # merging wordpiece embeddings
    result = []
    stack = []  # initialise stack with idx

    t_span = next(token_spans)
    for i in range(len(wp_e)):
        wp_span = inputs.token_to_chars(i)
        if is_subspan(wp_span, t_span): stack.append(i)
        else: 
            t_e = _merge_embeddings(wp_e, stack)
            result.append(t_e)
            t_span = next(token_spans) # if error is thrown, sth is wrong as every wp should be a subspan of some token

            assert is_subspan(wp_span, t_span)
            stack = [i]  # initialise stack with current idx
    
    # clear remaining accumulated tensors
    t_e = _merge_embeddings(wp_e, stack)
    result.append(t_e)

    result = torch.stack(result, dim = 0)
    assert len(result) == len(string.split()), f"{len(result)} != {len(string.split())}"
    return result

def merge_wordpiece_embeddings_by_word_ids(embeddings: Tensor, word_ids: List) -> Tensor:
    """ Merges wordpiece embeddings by averaging them. 
    Note: word_ids don't work as they cannot retrieve original tokens for some tokens (e.g. "l'argent")
    """

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
    # TODO: return list indicating alignment

    annotation_id: str = ann.annotation_id
    evidences: List[List[Evidence]] = ann.evidences
    label: str = str(ann.classification)
    query: str = ann.query 

    docids: List[str] = [f"{annotation_id}_hypothesis", f"{annotation_id}_premise"] # only for esnli
    # docids_2: List[str] = sorted(list(set([evclause.docid for evgroup in evidences for evclause in evgroup])))  # easily overfit esnli: contradiction both premise and hypothesis, neutral only hypothesis.
    # assert docids == docids_2, f"{docids} != {docids_2}"

    document_evidence_map: Dict[str, List[Tuple[int, int]]] = generate_document_evidence_map(evidences)
    assert set(document_evidence_map.keys()).issubset(set(docids)), "Evidence should come from docids!"

    token_embeddings = []
    rationale = []
    # TODO: kept_tokens doesn't make sense from backprop persp - deterministic alteration to op fn.
    # kept_tokens = []

    for docid in docids:
        # get token embeddings
        t_e = get_token_embeddings(docs[docid], tokenizer, embedding_model, merge_strategy="average")
        token_embeddings.extend(t_e)
        
        # get rationale
        r = [0.0] * len(t_e)
        if docid in document_evidence_map:
            for s, e in document_evidence_map[docid]: 
                r[s:e] = [1.0] * (e - s)

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

    return token_embeddings, rationale, label, annotation_id

def create_wp_instance(ann: Annotation, docs: Dict[str, str], tokenizer, embedding_model, logger=None):
    """ Note: This creates wordpiece not token embeddings. No longer supported as it's difficult to align WP embeddings.  """
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
        r = [0.0] * wp_e.size()[0]
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
                    r[i] = 1.0 
                    is_token_in_evd = reduce(lambda prev, evd: prev or tokens[i] in evd, document_evidence_str_map[docid], False)
                    if not is_token_in_evd and tokens[i] != '[UNK]':
                        logger.warning(f"{tokens[i]} is not found in evidence {document_evidence_str_map}! annotation_id: {annotation_id}")
                        logger.warning(tokens)
                        # raise ValueError
                    # if not is_token_in_evd: logger.warn(f"{tokens[i]} is not found in evidence {document_evidence_str_map}!")
                else:
                    r[i] = 0.0

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
    """ Follows ERASER paper's implementation. Returns macro token f1 score. """
    assert mask_pad.size() == rationale_pad.size()  # (L, N) == (L, N)
    running_scores = torch.zeros(3)
    prf_metric = PRFScore("binary")
    for i in range(len(token_lengths)):
        mask_instance = mask_pad[:token_lengths[i], i]
        rationale_instance = rationale_pad[:token_lengths[i], i]
        # print(f"mask instance: {mask_instance}")
        # print(f"rat instance: {rationale_instance}")
        assert len(mask_instance) == len(rationale_instance)

        scores = prf_metric(rationale_instance, mask_instance)
        running_scores += torch.tensor(scores)

    p, r, f1 = running_scores / len(token_lengths)
    return p, r, f1

def top_k_idxs_multid(a, k):
    """ https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor """
    v, i = torch.topk(a.flatten(), k)
    res = torch.tensor(np.unravel_index(i.numpy(), a.shape)).T
    return res

def get_top_k_prob_mask(prob_mask, k):
    """ Returns new tensor with top k most confident elements in mask retained. Rest are -1. """
    prob_mask_flat = prob_mask.flatten()
    conf_masks = torch.abs(prob_mask_flat - 0.5)
    v, i = torch.topk(conf_masks, k)
    res_flat = torch.full(prob_mask_flat.size(), -1.)
    res_flat[i] = prob_mask_flat[i]
    res = res_flat.view(prob_mask.size())
    
    return res

def parse_alignment(algn: str, reverse=False) -> Dict[int, List[int]]:
    """ Returns dictionary whose key i and sorted array of values j corresponds to the i-j Pharaoh format of input alignment.
    """
    op = {}
    was = algn.split()
    for wa in was:
        i, j = wa.split("-")
        i, j = int(i), int(j)
        if reverse:
            if j not in op:
                op[j] = [i]
            else:
                op[j].append(i)
                op[j].sort()
        else:
            if i not in op:
                op[i] = [j]
            else:
                op[i].append(j)
                op[i].sort()

    return op

def add_offsets(wa: Dict[int, List[int]], key_offset, val_offset):
    res = {}
    for k, v in wa.items():
        res[k + key_offset] = [x+val_offset for x in v]
    return res

def instantiate_models(config, device, enc_weights_fp=None, gen_weights_fp=None):
    enc = Encoder(config["encoder"]).to(device)
    gen = Generator(config["generator"]).to(device)
    if enc_weights_fp: enc.load_state_dict(torch.load(enc_weights_fp))
    if gen_weights_fp: gen.load_state_dict(torch.load(gen_weights_fp))
    return enc, gen

class PRFScore:
    """
    Class for precision, recall, f1 scores in Pytorch.
    """

    def __init__(self, average: str = 'macro', pos_label: int = 1):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        self.pos_label = pos_label
        if average not in [None, 'macro', 'weighted', 'binary']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        raise NotImplementedError
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_prf_count_for_label(labels: torch.Tensor, predictions: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate precision, recall, f1 and true count for the label

        Args:
            labels: tensor with original labels
            predictions: tensor with predictions
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return precision, recall, f1, true_count

    def __call__(self, labels: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        assert labels.dim() == 1, "Flatten labels first!"
        assert predictions.dim() == 1, "Flatten predictions first!"

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(labels, predictions)
        if self.average == 'binary':
            p, r, f1, _ = self.calc_prf_count_for_label(labels, predictions, self.pos_label)
            return p, r, f1

        scores = torch.zeros(3)
        for label_id in range(0, len(labels.unique())):
            p, r, f1, true_count = self.calc_prf_count_for_label(labels, predictions, label_id)

            if self.average == 'weighted':
                scores += torch.tensor([p, r, f1]) * true_count
            elif self.average == 'macro':
                scores += torch.tensor([p, r, f1])

        if self.average == 'weighted':
            scores = scores / len(labels)
        elif self.average == 'macro':
            scores = scores / len(labels.unique())

        return scores[0], scores[1], scores[2]

### TESTS

def test_get_wordpiece_embeddings():
    from transformers import AutoModel
    from transformers import AutoTokenizer
    """ Tests for contextual embeddings. Follows https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#2-input-formatting. """

    from scipy.spatial.distance import cosine

    # model_name = 'bert-base-uncased'
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
    model.eval()
    
    # english
    en = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    inputs = tokenizer(en, return_tensors="pt", truncation=True, add_special_tokens=False)
    we_en = get_wordpiece_embeddings(inputs, model)

    tokens = tokenizer.tokenize(en, truncation=True, add_special_tokens=False)
    assert len(we_en) == len(tokens)
    ids = [i for i, x in enumerate(tokens) if x == "bank"]

    same_bank = 1 - cosine(we_en[ids[0]], we_en[ids[1]])
    diff_bank = 1 - cosine(we_en[ids[0]], we_en[ids[2]])
    print(f"diff_bank vs same_bank for en: {diff_bank} vs {same_bank}")
    assert same_bank > diff_bank

def test_get_token_embeddings():
    from transformers import AutoModel
    from transformers import AutoTokenizer
    """ Tests for contextual embeddings. Follows https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#2-input-formatting. """

    from scipy.spatial.distance import cosine

    # model_name = 'bert-base-uncased'
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
    model.eval()
    
    # english
    en = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    t_en = get_token_embeddings(en, tokenizer, model, merge_strategy="first")

    ids = [i for i, x in enumerate(en.split()) if "bank" in x]

    same_bank = 1 - cosine(t_en[ids[0]], t_en[ids[1]])  # cosine similarity
    diff_bank = 1 - cosine(t_en[ids[0]], t_en[ids[2]])
    print(f"diff_bank vs same_bank for en: {diff_bank} vs {same_bank}")
    assert same_bank > diff_bank

    inputs = tokenizer(en, return_tensors="pt", truncation=True, add_special_tokens=False)
    we_en = get_wordpiece_embeddings(inputs, model)

    tokens = tokenizer.tokenize(en, truncation=True, add_special_tokens=False)
    assert len(we_en) == len(tokens)
    we_ids = [i for i, x in enumerate(tokens) if x == "bank"]
    # token and wordpiece embeddings should be same since "bank" is not split further.
    assert torch.equal(t_en[ids[0]], we_en[we_ids[0]])
    assert torch.equal(t_en[ids[1]], we_en[we_ids[1]])
    assert torch.equal(t_en[ids[2]], we_en[we_ids[2]])

def test_merge_wordpiece_embeddings_by_word_ids():
    embeddings = torch.rand(26, 13, 768)
    word_ids = [None, 0, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, None]
    merged_embeddings = merge_wordpiece_embeddings_by_word_ids(embeddings, word_ids)
    assert merged_embeddings.size()[0] == len(set(word_ids)) + 1  # +1 to account for None

    embeddings = torch.tensor([[0., 0.], [1., -1.], [2., -2.], [3., -3.]])  # 4x2
    word_ids = [None, 0, 0, None]
    merged_embeddings = merge_wordpiece_embeddings_by_word_ids(embeddings, word_ids)
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

def test_top_k_idxs_multid():
    a = torch.tensor([[4, 9, 7, 4, 0],
        [8, 1, 3, 1, 0],
        [9, 8, 4, 4, 8],
        [0, 9, 4, 7, 8],
        [8, 8, 0, 1, 4]])
    idxs = top_k_idxs_multid(a, 3)
    expected = torch.tensor([[3, 1],
        [2, 0],
        [0, 1]])
    assert torch.equal(idxs, expected), idxs

def test_get_top_k_prob_mask():
    prob_mask = torch.tensor([[0.7, 0.4], [0.1, 0.5]])
    top_k_prob_mask = get_top_k_prob_mask(prob_mask, 2)
    expected = torch.tensor([[0.7, -1.], [0.1, -1.]])
    print(expected.type())
    print(top_k_prob_mask.type())
    assert torch.equal(top_k_prob_mask, expected), f"{top_k_prob_mask} != {expected}"

def test_prfscore():
    from sklearn.metrics import f1_score
    for _ in range(1):
        labels = torch.randint(0, 10, (4096, 100)).flatten()
        predictions = torch.randint(0, 10, (4096, 100)).flatten()

        # TODO: binary test
        for av in ['macro', 'weighted']:
            my_p, my_r, my_f1 = PRFScore(av)(labels, predictions)
            
            p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average=av)
            e_f1 = f1_score(labels, predictions, average=av)
            assert np.isclose(my_p.item(), p)
            assert np.isclose(my_r.item(), r)
            assert np.isclose(my_f1.item(), f1)
            assert np.isclose(my_f1.item(), e_f1)
        
        labels = torch.randint(0, 2, (4096, 100)).flatten()
        predictions = torch.randint(0, 2, (4096, 100)).flatten()
        prf_metric = PRFScore("binary")
        my_p, my_r, my_f1 = prf_metric(labels, predictions)
        
        p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
        assert np.isclose(my_p.item(), p)
        assert np.isclose(my_r.item(), r)
        assert np.isclose(my_f1.item(), f1)




if __name__ == "__main__":
    print("Running unit tests...")
    # test_merge_character_spans()
    # test_merge_wordpiece_embeddings_by_word_ids()
    # test_gen_loss()
    test_score_hard_rationale_predictions()
    # test_top_k_idxs_multid()
    test_prfscore()
    # test_get_top_k_prob_mask()
    # test_get_wordpiece_embeddings()
    # test_get_token_embeddings()
    print("Unit tests passed!")


