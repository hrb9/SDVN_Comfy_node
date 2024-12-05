from .node.load import NODE_CLASS_MAPPINGS as load_node, NODE_DISPLAY_NAME_MAPPINGS as load_dis
from .node.merge import NODE_CLASS_MAPPINGS as merge_node, NODE_DISPLAY_NAME_MAPPINGS as merge_dis
from .node.creative import NODE_CLASS_MAPPINGS as creative_node, NODE_DISPLAY_NAME_MAPPINGS as creative_dis
from .node.chatbot import NODE_CLASS_MAPPINGS as chatbot_node, NODE_DISPLAY_NAME_MAPPINGS as chatbot_dis

NODE_CLASS_MAPPINGS = {
    **load_node,
    **merge_node,
    **creative_node,
    **chatbot_node,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **load_dis,
    **merge_dis,
    **creative_dis,
    **chatbot_dis,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
