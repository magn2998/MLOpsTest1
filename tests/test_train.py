import tests
import torch
import Test1.models.model as MyModels
import logging

LOGGER = logging.getLogger(__name__)

def test_train():
    randomData = torch.rand((10, 28*28))
    model = MyModels.MyCnnNetwork()
    pred, _ = model.forward(randomData)

    assert pred.shape == torch.Size((10,10)), "Shape of the output of the prediction should be [batchSize, 10]"
