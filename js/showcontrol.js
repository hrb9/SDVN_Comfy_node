import { app } from "../../scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

const doesInputWithNameExist = (node, name) => {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
};

const HIDDEN_TAG = "sdvnhide";

function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
            
    // Store the original properties of the widget if not already stored
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }       
        
    const origSize = node.size;

    // Set the widget type and computeSize based on the show flag
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    
    // Recursively handle linked widgets if they exist
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));
        
    // Calculate the new height for the node based on its computeSize method
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}   

function sdvnshowcontrol(node) {
    if (node.comfyClass === "SDVN Auto Generate") {
        const show = findWidgetByName(node, "AdvSetting")?.value === true;
        ["cfg", "sampler_name", "scheduler", "FluxGuidance"].forEach(name => toggleWidget(node, findWidgetByName(node, name), show));
    }
    if (node.comfyClass === "SDVN Load Image Ultimate") {
        const show = findWidgetByName(node, "mode")?.value === "Input folder";
        ["image"].forEach(name => toggleWidget(node, findWidgetByName(node, name), show));
    }
    if (node.comfyClass === "SDVN Load Image Ultimate") {
        const show = findWidgetByName(node, "mode")?.value ===  "Custom folder";
        ["folder_path","number_img"].forEach(name => toggleWidget(node, findWidgetByName(node, name), show));
    }
    if (node.comfyClass === "SDVN Load Image Ultimate") {
        const show = findWidgetByName(node, "mode")?.value ===  "Url";
        ["url"].forEach(name => toggleWidget(node, findWidgetByName(node, name), show));
    }
    if (node.comfyClass === "SDVN Load Image Ultimate") {
        const show = findWidgetByName(node, "mode")?.value ===  "Pintrest";
        ["pin_url", "range", "number", "random"].forEach(name => toggleWidget(node, findWidgetByName(node, name), show));
    }
    if (node.comfyClass === "SDVN Load Image Ultimate") {
        const show = findWidgetByName(node, "mode")?.value ===  "Insta";
        ["insta_url", "index"].forEach(name => toggleWidget(node, findWidgetByName(node, name), show));
    }
    if (node.comfyClass === "SDVN Save Text") {
        const show = findWidgetByName(node, "save_dir")?.value ===  "custom";
        ["custom_dir"].forEach(name => toggleWidget(node, findWidgetByName(node, name), show));
    }
}

function hookWidgetCallbacks(node) {
    for (const w of node.widgets || []) {
        if (typeof w.callback === "function" && !w.__sdvn_callback_wrapped) {
            const originalCallback = w.callback;
            w.callback = function (...args) {
                // Gọi lại callback gốc, giữ đúng ngữ cảnh this
                if (typeof originalCallback === "function") {
                    originalCallback.apply(this, args);
                }
                // Gọi lại hàm kiểm tra hiển thị
                sdvnshowcontrol(node);
            };
            w.__sdvn_callback_wrapped = true;
        }
    }
}

app.registerExtension({
    name: "SDVN.ShowControl",
    nodeCreated(node) {
        if (!node.comfyClass.startsWith("SDVN")) return;
        sdvnshowcontrol(node);
        hookWidgetCallbacks(node);
    }
});