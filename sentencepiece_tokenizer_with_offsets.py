import sys
sys.path.insert(0, "G:/krantirk/ALBERT/sentencepiece-pb2/")
import os
import sentencepiece as spm
import sentencepiece_pb2
import transformers

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path, "spiece.model"))
    
    def encode(self, sentence):
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))
        return tokens, offsets
spt = SentencePieceTokenizer("G:/krantirk/ALBERT/albert/")
spt.encode("hi, how are you?")

spt_trans=transformers.AlbertTokenizer.from_pretrained("G:/krantirk/ALBERT/albert/")
spt_trans.encode("hi, how are you?")

spt_trans.encode_plus("hi, how are you?","i am good")


spt.encode("positive negative neutral")

sp = spm.SentencePieceProcessor()
sp.load( "G:/krantirk/ALBERT/albert/spiece.model")
sent_ids=sp.encode_as_ids("hi,how are you?")
sent_peices=sp.encode_as_pieces("Explanation Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove th...")
" ".join(sent_peices).replace("_"," ")

