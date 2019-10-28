from .extension import MXNetExtension
from .config import Config
from openml.extensions import register_extension

__all__ = ['MXNetExtension', 'Config']

register_extension(MXNetExtension)
