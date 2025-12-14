import numpy as np
import pytest
import torch

from multimodal_ai.models.fusion_module import FusionEmbeddings, MultimodalMLP


def test_fusion_embeddings_shape(fake_embeddings, input_dims):
    """Verifies that fusion output dimensions match expectations (2560 + 768 = 3328)."""
    img, txt = fake_embeddings
    fuser = FusionEmbeddings()
    result = fuser.fuse_embeddings(img, txt)
    expected_dim = input_dims["img_dim"] + input_dims["txt_dim"]
    assert result.shape == (input_dims["n_samples"], expected_dim)
    assert result.dtype == np.float32


def test_fusion_error(input_dims):
    """Ensures ValueError is raised when image and text sample counts differ."""
    fuser = FusionEmbeddings()
    img = np.random.rand(10, input_dims["img_dim"])
    txt = np.random.rand(9, input_dims["txt_dim"])
    with pytest.raises(ValueError) as excinfo:
        fuser.fuse_embeddings(img, txt)
    assert "Fusion Error" in str(excinfo.value)


def test_mlp_forward_pass(fused_tensor, input_dims):
    """Validates that the MLP accepts the input tensor and outputs correct class logits."""
    total_dim = input_dims["img_dim"] + input_dims["txt_dim"]
    model = MultimodalMLP(input_dim=total_dim, num_classes=input_dims["n_classes"])
    output = model(fused_tensor)
    assert output.shape == (input_dims["n_samples"], input_dims["n_classes"])
    assert not torch.isnan(output).any()
