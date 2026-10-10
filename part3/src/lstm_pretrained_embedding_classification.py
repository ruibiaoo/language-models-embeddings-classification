import math
import re
import numpy as np
import pickle

from nltk.lm import KneserNeyInterpolated, Vocabulary
from nltk.util import everygrams
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from time import perf_counter
from data import prepare_dataset
from collections import Counter, defaultdict
from pathlib import Path