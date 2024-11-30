from .load import NODE_CLASS_MAPPINGS as load_node, NODE_DISPLAY_NAME_MAPPINGS as load_dis
from .merge import NODE_CLASS_MAPPINGS as merge_node, NODE_DISPLAY_NAME_MAPPINGS as merge_dis

NODE_CLASS_MAPPINGS = {
    **load_node,
    **merge_node,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **load_dis,
    **merge_dis,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
