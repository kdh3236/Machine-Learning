import gzip
import html
import os
from functools import lru_cache

import ftfy # Unicode 텍스트 정리 라이브러리
import regex as re

# BPE Tokenizer: 모든 Word를 Subword 단위로 분할하고, 가장 많이 등장하는 Word pair는 병합하여 Vocabulary를 구성

@lru_cache()
def default_bpe(): # BPE Tokniezr를 위한 기본 BPE 경로 반환
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz") # bpe_simple_vocab_16e6.txt.gz: BPE 룰과 Vocab이 포함된 파일

# Text: Byte sequence 
# 0~255 사이의 Byte 값 -> Unicode 문자로 매핑

# Data를 Web-crowling하여 Dataset에는 이모지, 깨진 Encoding 등이 포함될 수 있음
# Byte -> Unicode 매핑을 통해 위 문제를 해결

# BPE는 Unicode 문자열에서 작동하므로, Byte와 Unicode 간의 매핑이 필요
# 기존 BPE는 모르는 단어를 <UNK> 토큰으로 대체
# Unicode 문자 집합이 충분히 크다면, 모든 Byte가 커버되기 때문에 UNK 토큰을 피할 수 있음
@lru_cache()
def bytes_to_unicode(): # 각 Byte를 Unicode 문자로 매핑하는 함수
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word): # 한 Word에서 byte pair를 추출하는 함수
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text) # 깨진 Unicode 텍스트 수정
    text = html.unescape(html.unescape(text)) # HTML 엔티티를 실제 문자로 변환
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text) # \s+: 하나 이상의 공백 문자를 -> ' ': 단일 공백 문자로 변환
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode() # Byte to Unicode
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()} # Unicode to Byte
        
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges] # 어떤 Word pair를 병합할지 정의 / Tuple 형태로 변환
        
        vocab = list(bytes_to_unicode().values()) # Unicode 문자 목록
        vocab = vocab + [v+'</w>' for v in vocab] # '</w>': 단어의 끝을 나타내는 특수 토큰 / Unicode word에 Special token 추가
        for merge in merges:
            vocab.append(''.join(merge)) # merge는 ('e', 'r') 형태 -> ''join(merge): 'er' 형태로 변환하여 vocab에 추가
        vocab.extend(['<|startoftext|>', '<|endoftext|>']) # 문장 시작, 끝 토큰 추가
        
        self.encoder = dict(zip(vocab, range(len(vocab)))) # Token -> ID 매핑
        self.decoder = {v: k for k, v in self.encoder.items()} # ID -> Token 매핑

        # Merge 시, 우선 순위가 높은 Pair 먼저 병합되도록 한다.
        self.bpe_ranks = dict(zip(merges, range(len(merges)))) # Byte pair merge 순서를 명시
        
        # Chahe: 한 단어를 BPE로 변환한 결과를 저장, 재등장 시 Cache에서 불러옴
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}

        # 문장을 Word 단위로 분할하기 위한 정규 표현식 패턴
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        # Cache에 있으면 Cache에서 가져옴
        if token in self.cache:
            return self.cache[token]
        
        # 단어의 끝을 나타내는 특수 토큰 '</w>' 추가
        word = tuple(token[:-1]) + ( token[-1] + '</w>',) # (1, 2, 3) + (4, ) = (1, 2, 3, 4,)

        # 모든 가능한 Pair 추출
        pairs = get_pairs(word)

        # 한 단어인 경우, Single token + '</w>' 반환
        if not pairs:
            return token+'</w>'

        while True:
            # bpe_simple_vocab_16e6.txt.gz에 정의된 Merge 규칙에 대응되는 Pair 중, 가장 우선 순위가 높은 Pair 선택
            # get() 결과 값이 작으면 우선 순위가 높음
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i) # i의 위치부터 처음으로 나타나는 first의 인덱스 찾기
                    new_word.extend(word[i:j]) # Merge하지 않아야 하는 Pair 전까지의 문자들을 new_word에 추가 
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second: # Merge 대상 Pair인 경우
                    new_word.append(first+second) # Merge된 Pair를 new_word에 추가 / ex) 'r' + 'e' -> 're'를 그냥 추가
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word) # 리스트 -> 튜플로 변환
            word = new_word # Merge된 결과

            # While문 더 이상 Merger할 Pair가 없는 경우 종료
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8')) # 텍스트를 Byte 단위로 변환 후, Unicode로 매핑
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')) # BPE 토큰을 ID로 변환
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens]) # Token ID -> Token 변환 후, 하나의 Text로 결합
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text