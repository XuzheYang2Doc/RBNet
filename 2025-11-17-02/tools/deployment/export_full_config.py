import mmengine

mmengine.Config.fromfile("configs/my_model_configs/deeplabv3plus_all.py").dump("My_config.py")