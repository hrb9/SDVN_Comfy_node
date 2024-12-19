from PIL import Image
from PIL.ExifTags import TAGS
from PIL.PngImagePlugin import PngInfo
import json,ast

class img_info:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Get img_path by 'SDVN Load Image' node"}),
                "info_type": (["name","img_type","img_format", "color_mode", "image_size", "dpi", "metadata", "exif_data"],)
            }
        }

    CATEGORY = "📂 SDVN/🏞️ Image"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "read"

    def read(self, img_path, info_type):
        try:
            with Image.open(img_path) as img:
                img_name = img_path.split('/')[-1].split(".")[0]
                img_type = img_path.split('/')[-1].split(".")[-1]
                metadata = img.info
                color_mode = img.mode
                image_size = img.size
                image_format = img.format 
                dpi = img.info.get("dpi", None)
                exif_data = img._getexif()
                r =  {  "name": img_name,
                        "img_type": img_type,
                        "img_format": image_format,
                        "color_mode": color_mode,
                        "image_size": image_size,
                        "dpi": dpi,
                        "metadata": metadata,
                        "exif_data": exif_data}
                return (str(r[info_type]),)
        except:
            print(f'Wrong path: {img_path}')
            return ("No data",)
        
def node_list(workflow):
    workflow = json.loads(workflow)
    node_list = []
    for i in workflow["nodes"]:
        node_list += [i["type"]] if "type" in i else ""
    return "\n".join(node_list)

class metadata_check:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metadata": ("STRING",{"forceInput": True}),
                "info_type": (["ComfyUI_Workflow_Json", "ComfyUI_Node_List", "Automatic_Info", "Automatic_Positive", "Automatic_Negative", "Automatic_Setting",],)
            }
        }

    CATEGORY = "📂 SDVN/🏞️ Image"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "read"

    def read(self, metadata, info_type):
        if metadata == "No data":
            return (metadata,)
        else:
            try:
                data_dic = ast.literal_eval(metadata)
            except:
                return ("No data",)
            r = {}
            if "ComfyUI" in info_type and "workflow" in data_dic:
                    r["ComfyUI_Workflow_Json"] = data_dic["workflow"]
                    r["ComfyUI_Node_List"] = node_list(data_dic["workflow"])
            if "Automatic" in info_type and "parameters" in data_dic:
                if len(data_dic["parameters"].split("\n")) == 3:
                    r["Automatic_Info"] = data_dic["parameters"]
                    r["Automatic_Positive"] = r["Automatic_Info"].split("\n")[0]
                    r["Automatic_Negative"] = r["Automatic_Info"].split("\n")[1].split('Negative prompt:')[-1].strip()
                    r["Automatic_Setting"] = r["Automatic_Info"].split("\n")[2]
            try:
                resulf = str(r[info_type])
            except:
                resulf = "No data"
            return (resulf,)

exif_tags = {
    'InteropIndex': 1,
    'InteropVersion': 2,
    'ProcessingSoftware': 11,
    'NewSubfileType': 254,
    'SubfileType': 255,
    'ImageWidth': 256,
    'ImageLength': 257,
    'BitsPerSample': 258,
    'Compression': 259,
    'PhotometricInterpretation': 262,
    'Thresholding': 263,
    'CellWidth': 264,
    'CellLength': 265,
    'FillOrder': 266,
    'DocumentName': 269,
    'ImageDescription': 270,
    'Make': 271,
    'Model': 272,
    'StripOffsets': 273,
    'Orientation': 274,
    'SamplesPerPixel': 277,
    'RowsPerStrip': 278,
    'StripByteCounts': 279,
    'XResolution': 282,
    'YResolution': 283,
    'PlanarConfiguration': 284,
    'ResolutionUnit': 296,
    'TransferFunction': 301,
    'Software': 305,
    'DateTime': 306,
    'Artist': 315,
    'WhitePoint': 318,
    'PrimaryChromaticities': 319,
    'SubIFDs': 330,
    'JPEGInterchangeFormat': 513,
    'JPEGInterchangeFormatLength': 514,
    'YCbCrCoefficients': 529,
    'YCbCrSubSampling': 530,
    'YCbCrPositioning': 531,
    'ReferenceBlackWhite': 532,
    'Copyright': 33432,
    'ExifOffset': 34665,
    'GPSInfo': 34853,
    'ISOSpeedRatings': 34855,
    'SensitivityType': 34864,
    'StandardOutputSensitivity': 34865,
    'RecommendedExposureIndex': 34866,
    'ExifVersion': 36864,
    'DateTimeOriginal': 36867,
    'DateTimeDigitized': 36868,
    'OffsetTime': 36880,
    'OffsetTimeOriginal': 36881,
    'OffsetTimeDigitized': 36882,
    'ComponentsConfiguration': 37121,
    'CompressedBitsPerPixel': 37122,
    'ShutterSpeedValue': 37377,
    'ApertureValue': 37378,
    'BrightnessValue': 37379,
    'ExposureBiasValue': 37380,
    'MaxApertureValue': 37381,
    'SubjectDistance': 37382,
    'MeteringMode': 37383,
    'LightSource': 37384,
    'Flash': 37385,
    'FocalLength': 37386,
    'SubjectArea': 37396,
    'MakerNote': 37500,
    'UserComment': 37510,
    'SubSecTime': 37520,
    'SubSecTimeOriginal': 37521,
    'SubSecTimeDigitized': 37522,
    'FlashpixVersion': 40960,
    'ColorSpace': 40961,
    'PixelXDimension': 40962,
    'PixelYDimension': 40963,
    'RelatedSoundFile': 40964,
    'InteroperabilityOffset': 40965,
    'FocalPlaneXResolution': 41486,
    'FocalPlaneYResolution': 41487,
    'FocalPlaneResolutionUnit': 41488,
    'SubjectLocation': 41492,
    'ExposureIndex': 41493,
    'SensingMethod': 41495,
    'FileSource': 41728,
    'SceneType': 41729,
    'CFAPattern': 41730,
    'CustomRendered': 41985,
    'ExposureMode': 41986,
    'WhiteBalance': 41987,
    'DigitalZoomRatio': 41988,
    'FocalLengthIn35mmFilm': 41989,
    'SceneCaptureType': 41990,
    'GainControl': 41991,
    'Contrast': 41992,
    'Saturation': 41993,
    'Sharpness': 41994,
    'SubjectDistanceRange': 41996,
    'ImageUniqueID': 42016,
    'CameraOwnerName': 42032,
    'BodySerialNumber': 42033,
    'LensSpecification': 42034,
    'LensMake': 42035,
    'LensModel': 42036,
    'LensSerialNumber': 42037
}

class exif_check:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "exif": ("STRING",{"forceInput": True}),
                "info_type": (list(exif_tags),)
            }
        }

    CATEGORY = "📂 SDVN/🏞️ Image"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "read"

    def read(self, exif, info_type):
        if exif == "No data":
            return (exif,)
        else:
            try:
                data_dic = ast.literal_eval(exif)
            except:
                return ("No data",)
            try:
                resulf = str(data_dic[exif_tags[info_type]])
            except:
                resulf = "No data"
            return (resulf,)

NODE_CLASS_MAPPINGS = {
    "SDVN Image Info": img_info,
    "SDVN Metadata Check": metadata_check,
    "SDVN Exif check": exif_check,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDVN Image Info": "ℹ️ Image Info",
    "SDVN Metadata Check": "ℹ️ Metadata check",
    "SDVN Exif check": "ℹ️ Exif check"
}
