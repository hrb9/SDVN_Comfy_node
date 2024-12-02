import re


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class Easy_IPA_weight:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SDXL": ("BOOLEAN", {"default": False},),
            "Weight": ("STRING", {"default": "0:1,1:1,1,1,4-15:1", "multiline": False, })
        }
        }

    CATEGORY = "✨ SDVN"

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = (
        "Ex: 0-4:1,6:1,1,1 or 0-15:1 or 1,1,1,1,1 or 1:1,5:1,7:1",)
    FUNCTION = "convert_wright"

    def convert_wright(self, SDXL, Weight):
        max_block = 10 if SDXL else 15
        Weight = Weight.split(",")
        index = 0
        convert = []
        for i in range(len(Weight)):
            if ':' not in Weight[i]:
                convert += [f'{str(index)}:{Weight[i]}'] if index <= max_block else []
                index += 1
            elif '-' in Weight[i]:
                ran, num = Weight[i].split(':')
                min, max = ran.split('-')
                index = int(min)
                for j in range(int(min), int(max)+1):
                    convert += [f'{str(index)}:{num}'] if index <= max_block else []
                    index += 1
            else:
                convert += [Weight[i]] if index <= max_block else []
                index += 1
        final_weight = ",".join(convert)
        print(f'Block weight: [{final_weight}]')
        return (final_weight,)


class run_python_code:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (any,),
                "function": ("STRING", {"default": """
def function(input):
    output = input.strip()                               
    return output                       
                """, "multiline": True, })
            }
        }

    CATEGORY = "✨ SDVN/Dev"

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "python_function"

    def python_function(self, input, function):
        pattern = r"def.*?return[^\n]*"
        match = re.search(pattern, function, re.DOTALL)
        function = match.group(0) if match else ""
        pattern = r"def\s+(\w+)\s*\("
        matches = re.findall(pattern, function)[0]
        local_context = {}
        exec(function, {}, local_context)
        function = local_context[matches]
        output = function(input)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "SDVN Easy IPAdapter weight": Easy_IPA_weight,
    "SDVN Run Python Code": run_python_code,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Easy IPAdapter weight": "✨ IPAdapter weight",
    "SDVN Run Python Code": "👨🏻‍💻 Run Python Code"
}
