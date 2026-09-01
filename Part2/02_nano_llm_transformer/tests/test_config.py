from core.config import ModelConfig
def test_valid_config():ModelConfig(d_model=128,n_heads=4,n_kv_heads=2).validate()
