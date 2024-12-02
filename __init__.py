from .node.load import NODE_CLASS_MAPPINGS as load_node, NODE_DISPLAY_NAME_MAPPINGS as load_dis
from .node.merge import NODE_CLASS_MAPPINGS as merge_node, NODE_DISPLAY_NAME_MAPPINGS as merge_dis
from .node.easy import NODE_CLASS_MAPPINGS as easy_node, NODE_DISPLAY_NAME_MAPPINGS as easy_dis
from .node.chatbot import NODE_CLASS_MAPPINGS as chatbot_node, NODE_DISPLAY_NAME_MAPPINGS as chatbot_dis

NODE_CLASS_MAPPINGS = {
    **load_node,
    **merge_node,
    **easy_node,
    **chatbot_node,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **load_dis,
    **merge_dis,
    **easy_dis,
    **chatbot_dis,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
