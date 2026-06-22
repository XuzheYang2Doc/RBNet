import mmengine

mmengine.Config.fromfile("configs/my_model_configs/deeplabv3plus.py").dump("deeplabv3plus.py")