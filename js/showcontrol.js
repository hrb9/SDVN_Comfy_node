import { app } from "../../scripts/app.js";

const HIDDEN_TYPE = "sdvnhide";
const origProps = {};

function injectHidden(widget) {
    if (!widget) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            _type: widget.type,
            _computeSize: widget.computeSize || (() => [widget.width || 200, 20])
        };
    }

    widget.computeSize = () => widget.hidden ? [0, -4] : origProps[widget.name]._computeSize();
    Object.defineProperty(widget, "type", {
        configurable: true,
        get() {
            return widget.hidden ? HIDDEN_TYPE : origProps[widget.name]._type;
        },
        set(val) {
            origProps[widget.name]._type = val;
        }
    });
}

function findWidget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function toggleWidgets(node, showMap = {}) {
    for (const [name, show] of Object.entries(showMap)) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        injectHidden(widget);
        widget.hidden = !show;
    }
    node.setSize([node.size[0], node.computeSize()[1]]);
}

function applyVisibilityLogic(node) {
    try {
        const modeWidget = findWidget(node, "mode");
        const mode = modeWidget ? modeWidget.value : null;

        const advWidget = findWidget(node, "AdvSetting");
        const adv = advWidget ? advWidget.value : null;

        const saveDirWidget = findWidget(node, "save_dir");
        const saveDir = saveDirWidget ? saveDirWidget.value : null;

        if (node.comfyClass === "SDVN Auto Generate") {
            toggleWidgets(node, {
                cfg: adv === true,
                sampler_name: adv === true,
                scheduler: adv === true,
                FluxGuidance: adv === true,
                Upscale_model: adv === true
            });
        }

        if (node.comfyClass === "SDVN Load Image Ultimate") {
            toggleWidgets(node, {
                image: mode === "Input folder",
                folder_path: mode === "Custom folder",
                number_img: mode === "Custom folder",
                url: mode === "Url",
                pin_url: mode === "Pintrest",
                range: mode === "Pintrest",
                number: mode === "Pintrest",
                random: mode === "Pintrest",
                insta_url: mode === "Insta",
                index: mode === "Insta"
            });
        }

        if (node.comfyClass === "SDVN Save Text") {
            toggleWidgets(node, {
                custom_dir: saveDir === "custom"
            });
        }

        app.canvas?.draw(true);
    } catch (e) {
        console.error("[SDVN.ShowControl] Visibility logic failed:", e, node);
    }
}

function watchValueChange(widget, node) {
    if (widget && !widget.__watched) {
        try {
            const desc = Object.getOwnPropertyDescriptor(widget, "value");
            if (!desc || desc.configurable) {
                widget._value = widget.value;
                Object.defineProperty(widget, "value", {
                    get() {
                        return this._value;
                    },
                    set(val) {
                        this._value = val;
                        try {
                            applyVisibilityLogic(node);
                            app.canvas?.draw(true);
                        } catch (err) {
                            console.error("[SDVN.ShowControl] applyVisibilityLogic error:", err);
                        }
                    },
                    configurable: true
                });
            }
        } catch (err) {
            console.warn(`[SDVN.ShowControl] Không thể hook 'value' cho widget ${widget.name}`, err);
        }
        widget.__watched = true;
    }
}

function prepareWidgets(node) {
    (node.widgets || []).forEach(widget => {
        watchValueChange(widget, node);
    });
}

app.registerExtension({
    name: "SDVN.ShowControl",
    nodeCreated(node) {
        if (!node.comfyClass?.startsWith("SDVN")) return;
        prepareWidgets(node);
        applyVisibilityLogic(node);
    }
});