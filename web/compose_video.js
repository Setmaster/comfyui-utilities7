/**
 * Compose Video Node - Video Preview Extension
 * Adds video preview widget to display output after execution.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Utility to chain callbacks on an object property
 */
function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existent object");
        return;
    }
    if (property in object && object[property]) {
        const originalCallback = object[property];
        object[property] = function () {
            const result = originalCallback.apply(this, arguments);
            return callback.apply(this, arguments) ?? result;
        };
    } else {
        object[property] = callback;
    }
}

/**
 * Resize node to fit content
 */
function fitHeight(node) {
    node.setSize([
        node.size[0],
        node.computeSize([node.size[0], node.size[1]])[1]
    ]);
    node?.graph?.setDirtyCanvas(true);
}

/**
 * Add format-specific widgets that appear/disappear based on selected format
 * Also handles proper serialization/deserialization for workflow save/load
 */
function addFormatWidgets(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        // Find the format widget and format_widget_values widget
        let formatWidget = null;
        let formatWidgetIndex = -1;
        let formatValuesWidget = null;

        for (let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i].name === "format") {
                formatWidget = this.widgets[i];
                formatWidgetIndex = i + 1;
            } else if (this.widgets[i].name === "format_widget_values") {
                formatValuesWidget = this.widgets[i];
            }
        }

        if (!formatWidget) return;

        // Hide the format_widget_values widget (it's just for data transport)
        if (formatValuesWidget) {
            formatValuesWidget.type = "hidden";
            formatValuesWidget.computeSize = () => [0, -4];
        }

        let formatWidgetsCount = 0;
        let formatWidgetNames = []; // Track names of current format widgets
        const node = this;

        // Function to get current format widget values as JSON
        const getFormatWidgetValuesJSON = () => {
            const values = {};
            for (const name of formatWidgetNames) {
                const w = node.widgets.find(w => w.name === name);
                if (w) {
                    values[name] = w.value;
                }
            }
            return JSON.stringify(values);
        };

        // Make format_widget_values always return current values when serialized
        if (formatValuesWidget) {
            formatValuesWidget.serializeValue = () => getFormatWidgetValuesJSON();
            // Also override the value getter to always return current values
            Object.defineProperty(formatValuesWidget, 'value', {
                get: () => getFormatWidgetValuesJSON(),
                set: () => {}, // Ignore sets, we always compute the value
                configurable: true
            });
        }

        // Function to create format-specific widgets
        const updateFormatWidgets = (value, savedValues = null) => {
            // Get format-specific widgets from node definition
            const formats = LiteGraph.registered_node_types[node.type]
                ?.nodeData?.input?.required?.format?.[1]?.formats;

            let newWidgets = [];
            formatWidgetNames = [];

            if (formats?.[value]) {
                const formatWidgetDefs = formats[value];
                for (const wDef of formatWidgetDefs) {
                    let type = wDef[2]?.widgetType ?? wDef[1];
                    if (Array.isArray(type)) {
                        type = "COMBO";
                    }
                    // Create the widget
                    app.widgets[type](node, wDef[0], wDef.slice(1), app);
                    const w = node.widgets.pop();
                    w.config = wDef.slice(1);

                    // Apply tooltip if defined
                    const tooltip = wDef[2]?.tooltip;
                    if (tooltip) {
                        w.tooltip = tooltip;
                    }

                    // Restore saved value if available
                    if (savedValues && w.name in savedValues) {
                        w.value = savedValues[w.name];
                    }

                    newWidgets.push(w);
                    formatWidgetNames.push(w.name);
                }
            }

            // Remove old format widgets and insert new ones
            const removed = node.widgets.splice(formatWidgetIndex, formatWidgetsCount, ...newWidgets);

            // Clean up removed widgets
            for (const w of removed) {
                w?.onRemove?.();
            }

            fitHeight(node);
            formatWidgetsCount = newWidgets.length;
        };

        // Hook into format widget callback
        chainCallback(formatWidget, "callback", (value) => {
            updateFormatWidgets(value);
        });

        // Store reference to update function for use in onConfigure
        node._updateFormatWidgets = updateFormatWidgets;

        // Trigger initial format widget setup
        if (formatWidget.value) {
            updateFormatWidgets(formatWidget.value);
        }

        // Handle workflow load - restore format widgets with saved values
        chainCallback(node, "onConfigure", function (info) {
            // Use our custom saved format widget values if available
            const savedFormatValues = info._formatWidgetValues || {};

            // Find format value from standard widget restoration
            // (ComfyUI will have already restored the format widget value)
            setTimeout(() => {
                // Use setTimeout to run after ComfyUI's standard widget restoration
                const formatValue = formatWidget.value;
                if (formatValue && node._updateFormatWidgets) {
                    node._updateFormatWidgets(formatValue, savedFormatValues);
                }
            }, 0);
        });

        // Save format widget values separately (don't interfere with standard serialization)
        chainCallback(node, "onSerialize", function (info) {
            if (!node.widgets) return;
            // Save format-specific widgets to a custom property
            const formatValues = {};
            for (const name of formatWidgetNames) {
                const w = node.widgets.find(w => w.name === name);
                if (w) {
                    formatValues[name] = w.value;
                }
            }
            info._formatWidgetValues = formatValues;
        });
    });
}

app.registerExtension({
    name: "utilities7.ComposeVideo",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name !== "ComposeVideo") {
            return;
        }

        // Add format-specific widgets (audio_output, crf, etc.)
        addFormatWidgets(nodeType, nodeData);

        // Add video preview widget
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const node = this;

            // Create container element
            const container = document.createElement("div");
            container.style.width = "100%";

            // Create preview widget
            const previewWidget = this.addDOMWidget("videopreview", "preview", container, {
                serialize: false,
                hideOnZoom: false,
                getValue() {
                    return container.value;
                },
                setValue(v) {
                    container.value = v;
                },
            });

            // Initialize widget state
            previewWidget.value = {
                hidden: false,
                paused: false,
                muted: true,
                params: {}
            };

            // Compute size based on aspect ratio
            previewWidget.computeSize = function (width) {
                if (this.aspectRatio && !this.parentEl.hidden) {
                    let height = (node.size[0] - 20) / this.aspectRatio + 10;
                    if (!(height > 0)) {
                        height = 0;
                    }
                    this.computedHeight = height + 10;
                    return [width, height];
                }
                return [width, -4]; // No source loaded, hide widget
            };

            // Create parent element for video/image
            previewWidget.parentEl = document.createElement("div");
            previewWidget.parentEl.className = "compose_video_preview";
            previewWidget.parentEl.style.width = "100%";
            container.appendChild(previewWidget.parentEl);

            // Create video element
            previewWidget.videoEl = document.createElement("video");
            previewWidget.videoEl.controls = false;
            previewWidget.videoEl.loop = true;
            previewWidget.videoEl.muted = true;
            previewWidget.videoEl.style.width = "100%";

            // Handle video metadata loaded
            previewWidget.videoEl.addEventListener("loadedmetadata", () => {
                previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
                fitHeight(node);
            });

            // Handle video load error
            previewWidget.videoEl.addEventListener("error", () => {
                previewWidget.parentEl.hidden = true;
                fitHeight(node);
            });

            // Unmute on hover, mute on leave
            previewWidget.videoEl.onmouseenter = () => {
                previewWidget.videoEl.muted = previewWidget.value.muted;
            };
            previewWidget.videoEl.onmouseleave = () => {
                previewWidget.videoEl.muted = true;
            };

            // Create image element (for gif/webp)
            previewWidget.imgEl = document.createElement("img");
            previewWidget.imgEl.style.width = "100%";
            previewWidget.imgEl.hidden = true;
            previewWidget.imgEl.onload = () => {
                previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
                fitHeight(node);
            };

            previewWidget.parentEl.appendChild(previewWidget.videoEl);
            previewWidget.parentEl.appendChild(previewWidget.imgEl);

            // Forward all mouse events to canvas for proper interaction
            // This prevents the preview from blocking zoom, pan, and click events
            container.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                return app.canvas._mousedown_callback(e);
            }, true);
            container.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                return app.canvas._mousedown_callback(e);
            }, true);
            container.addEventListener("mousewheel", (e) => {
                e.preventDefault();
                return app.canvas._mousewheel_callback(e);
            }, true);
            container.addEventListener("pointermove", (e) => {
                e.preventDefault();
                return app.canvas._mousemove_callback(e);
            }, true);
            container.addEventListener("pointerup", (e) => {
                e.preventDefault();
                return app.canvas._mouseup_callback(e);
            }, true);

            // Update preview source
            previewWidget.updateSource = function () {
                if (!this.value.params || !this.value.params.filename) {
                    return;
                }

                const params = { ...this.value.params };
                this.parentEl.hidden = this.value.hidden;

                const formatType = params.format?.split("/")[0];
                const formatExt = params.format?.split("/")[1];

                // Build URL for ComfyUI's /view endpoint
                const viewParams = new URLSearchParams({
                    filename: params.filename,
                    subfolder: params.subfolder || "",
                    type: params.type || "output"
                });
                const url = api.apiURL("/view?" + viewParams.toString());

                if (formatType === "video" || formatExt === "gif") {
                    // Use video element for video formats and gifs
                    this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
                    this.videoEl.src = url;
                    this.videoEl.hidden = false;
                    this.imgEl.hidden = true;
                } else if (formatType === "image") {
                    // Use img element for static images (webp)
                    this.imgEl.src = url;
                    this.videoEl.hidden = true;
                    this.imgEl.hidden = false;
                }
            };

            // Method to update parameters and refresh preview
            this.updateParameters = (params, forceUpdate = false) => {
                if (!previewWidget.value.params) {
                    previewWidget.value.params = {};
                }
                Object.assign(previewWidget.value.params, params);
                if (forceUpdate) {
                    previewWidget.updateSource();
                }
            };
        });

        // Hook into onExecuted to update preview when node finishes
        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            if (message?.gifs && message.gifs.length > 0) {
                this.updateParameters(message.gifs[0], true);
            }
        });

        // Add right-click context menu options
        chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
            const previewWidget = this.widgets.find((w) => w.name === "videopreview");
            if (!previewWidget) return;

            const newOptions = [];

            // Get current preview URL
            let url = null;
            if (previewWidget.videoEl && !previewWidget.videoEl.hidden && previewWidget.videoEl.src) {
                url = previewWidget.videoEl.src;
            } else if (previewWidget.imgEl && !previewWidget.imgEl.hidden && previewWidget.imgEl.src) {
                url = previewWidget.imgEl.src;
            }

            if (url) {
                // Open in new tab
                newOptions.push({
                    content: "Open preview",
                    callback: () => {
                        window.open(url, "_blank");
                    },
                });

                // Download preview
                newOptions.push({
                    content: "Save preview",
                    callback: () => {
                        const a = document.createElement("a");
                        a.href = url;
                        a.setAttribute("download", previewWidget.value.params?.filename || "video");
                        document.body.append(a);
                        a.click();
                        requestAnimationFrame(() => a.remove());
                    },
                });

                // Copy file path
                if (previewWidget.value.params?.fullpath) {
                    newOptions.push({
                        content: "Copy output filepath",
                        callback: async () => {
                            try {
                                await navigator.clipboard.writeText(previewWidget.value.params.fullpath);
                            } catch (e) {
                                console.error("Failed to copy path:", e);
                            }
                        },
                    });
                }
            }

            // Pause/Resume (only for video element)
            if (previewWidget.videoEl && !previewWidget.videoEl.hidden) {
                const pauseLabel = previewWidget.value.paused ? "Resume preview" : "Pause preview";
                newOptions.push({
                    content: pauseLabel,
                    callback: () => {
                        if (previewWidget.value.paused) {
                            previewWidget.videoEl.play();
                        } else {
                            previewWidget.videoEl.pause();
                        }
                        previewWidget.value.paused = !previewWidget.value.paused;
                    },
                });
            }

            // Hide/Show preview
            const visLabel = previewWidget.value.hidden ? "Show preview" : "Hide preview";
            newOptions.push({
                content: visLabel,
                callback: () => {
                    if (!previewWidget.videoEl.hidden && !previewWidget.value.hidden) {
                        previewWidget.videoEl.pause();
                    } else if (previewWidget.value.hidden && !previewWidget.videoEl.hidden && !previewWidget.value.paused) {
                        previewWidget.videoEl.play();
                    }
                    previewWidget.value.hidden = !previewWidget.value.hidden;
                    previewWidget.parentEl.hidden = previewWidget.value.hidden;
                    fitHeight(this);
                },
            });

            // Mute/Unmute
            const muteLabel = previewWidget.value.muted ? "Unmute preview" : "Mute preview";
            newOptions.push({
                content: muteLabel,
                callback: () => {
                    previewWidget.value.muted = !previewWidget.value.muted;
                },
            });

            // Add separator if there are existing options
            if (options.length > 0 && options[0] != null && newOptions.length > 0) {
                newOptions.push(null);
            }

            // Prepend our options
            options.unshift(...newOptions);
        });
    },
});
