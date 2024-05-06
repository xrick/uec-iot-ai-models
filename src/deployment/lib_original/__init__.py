from .bytes import to_bytes, from_bytes, byte_conversion_tests, load_data, load_raw, save_raw, save_scores
from .constants import quant_support, crops, feature_count
from .quantize import quantization_tests, get_cast
from .opts import parse

def run_tests():
    byte_conversion_tests()
    quantization_tests()