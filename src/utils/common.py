# src/utils/common.py

class BaseModule:
    def __init__(self, conf: dict, module_key: str):
        """
        基础模块类，自动从全局配置字典conf中提取对应模块配置
        :param conf: 全局配置字典
        :param module_key: 当前模块配置key，如"embedding", "library"等
        """
        self.conf = conf
        self.module_conf = conf.get(module_key, {})
