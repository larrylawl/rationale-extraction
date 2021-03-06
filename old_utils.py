import json
import os
import unicodedata

from dataclasses import dataclass, asdict, is_dataclass
from itertools import chain
from typing import Dict, List, Set, Tuple, Union

@dataclass(eq=True, frozen=True)
class Evidence:
    """
    (docid, start_token, end_token) form the only official Evidence; sentence level annotations are for convenience.
    Args:
        text: Some representation of the evidence text
        docid: Some identifier for the document
        start_token: The canonical start token, inclusive
        end_token: The canonical end token, exclusive
        start_sentence: Best guess start sentence, inclusive
        end_sentence: Best guess end sentence, exclusive
    """
    text: Union[str, List[int]]
    docid: str
    start_token: int=-1
    end_token: int=-1
    start_sentence: int=-1
    end_sentence: int=-1

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

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
    
    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

def preprocess_line(line: str) -> str:
    line = unicodedata.normalize("NFKD", line) # load french spaces properly
    line = line.replace('\xad', '')  # replace soft-hypens in multirc
    line = line.strip()
    return line

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

def load_jsonl(fp: str) -> List[dict]:
    ret = []
    with open(fp, 'r', encoding='utf-8') as inf:
        for line in inf:
            line = unicodedata.normalize("NFKD", line)  # load french spaces properly
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
            line = unicodedata.normalize("NFKD", line)  # load french spaces properly
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

def load_documents(data_dir: str, docids: Set[str]=None) -> Dict[str, List[List[str]]]:
    """Loads a subset of available documents from disk.

    Each document is assumed to be serialized as newline ('\n') separated sentences.
    Each sentence is assumed to be space (' ') joined tokens.
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
        with open(os.path.join(docs_dir, d), 'r', encoding='utf-8') as inf:
            lines = [preprocess_line(l) for l in inf.readlines()]
            lines = list(filter(lambda x: bool(len(x)), lines))
            tokenized = [list(filter(lambda x: bool(len(x)), line.strip().split(' '))) for line in lines]
            res[d] = tokenized
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

def intern_documents(documents: Dict[str, List[List[str]]], word_interner: Dict[str, int], unk_token: str):
    """
    Replaces every word with its index in an embeddings file.

    If a word is not found, uses the unk_token instead
    """
    ret = dict()
    unk = word_interner[unk_token]
    for docid, sentences in documents.items():
        ret[docid] = [[word_interner.get(w, unk) for w in s] for s in sentences]
    return ret

def intern_annotations(annotations: List[Annotation], word_interner: Dict[str, int], unk_token: str):
    ret = []
    for ann in annotations:
        ev_groups = []
        for ev_group in ann.evidences:
            evs = []
            for ev in ev_group:
                evs.append(Evidence(
                    text=tuple([word_interner.get(t, word_interner[unk_token]) for t in ev.text.split()]),
                    docid=ev.docid,
                    start_token=ev.start_token,
                    end_token=ev.end_token,
                    start_sentence=ev.start_sentence,
                    end_sentence=ev.end_sentence))
            ev_groups.append(tuple(evs))
        ret.append(Annotation(annotation_id=ann.annotation_id,
                              query=tuple([word_interner.get(t, word_interner[unk_token]) for t in ann.query.split()]),
                              evidences=frozenset(ev_groups),
                              classification=ann.classification,
                              query_type=ann.query_type))
    return ret

def load_documents_from_file(data_dir: str, docids: Set[str]=None) -> Dict[str, List[List[str]]]:
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
        lines = documents[d].split('\n')
        tokenized = [line.strip().split(' ') for line in lines]
        res[d] = tokenized
    return res


def get_algn_grid(was, train_anns):
    """ Creates alignment mask of cross product indexes (i.e. index [i, j] indicates that there is an alignment for src token i of annotation j)
    cross pdt idx: https://numpy.org/doc/stable/reference/generated/numpy.ix_.html
    """

    """ Updates alignment mask with values in word alignment.
    [i, j] == k denotes the ith token of src aligns to the kth token of tgt for the jth annotation. 
    k == -1 if alignment does not exist.
    """
    algn_seq = [] 
    for i, ann in enumerate(train_anns):
        wa = parse_alignment(was[ann.annotation_id]["alignment"])
        for k in wa.keys(): algn_seq.extend([k, i])
    algn_grid = np.ix_(*algn_seq)
    return algn_grid