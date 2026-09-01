from core.tokenizer import ByteTokenizer
def test_round_trip_unicode():
    t=ByteTokenizer();s='Hello, café 👋';assert t.decode(t.encode(s))==s
