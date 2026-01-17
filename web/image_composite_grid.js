import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "comfyui-utilities.ImageCompositeGrid",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImageCompositeGrid") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, []);

                const imageAmountWidget = this.widgets?.find(w => w.name === "image_amount");
                if (!imageAmountWidget) return;

                this._imageInputCount = 0;

                const updateImageInputs = (newCount) => {
                    const currentCount = this._imageInputCount;

                    if (newCount > currentCount) {
                        for (let i = currentCount + 1; i <= newCount; i++) {
                            this.addInput(`image_${i}`, "IMAGE");
                        }
                    } else if (newCount < currentCount) {
                        for (let i = currentCount; i > newCount; i--) {
                            const inputName = `image_${i}`;
                            const inputIndex = this.inputs?.findIndex(inp => inp.name === inputName);
                            if (inputIndex !== undefined && inputIndex >= 0) {
                                this.removeInput(inputIndex);
                            }
                        }
                    }

                    this._imageInputCount = newCount;

                    const currentSize = this.size;
                    const minSize = this.computeSize();
                    this.setSize([
                        Math.max(currentSize[0], minSize[0]),
                        Math.max(currentSize[1], minSize[1])
                    ]);
                    app.graph.setDirtyCanvas(true, true);
                };

                const initialCount = imageAmountWidget.value || 4;
                updateImageInputs(initialCount);

                const originalCallback = imageAmountWidget.callback;
                imageAmountWidget.callback = (value, ...args) => {
                    updateImageInputs(value);
                    if (originalCallback) {
                        originalCallback.call(this, value, ...args);
                    }
                };
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, [info]);

                const imageAmountWidget = this.widgets?.find(w => w.name === "image_amount");
                if (!imageAmountWidget) return;

                const savedCount = imageAmountWidget.value || 4;

                if (this._imageInputCount === undefined) {
                    this._imageInputCount = 0;
                }

                const existingImageInputs = this.inputs?.filter(inp =>
                    inp.name && inp.name.startsWith("image_")
                ).length || 0;

                this._imageInputCount = existingImageInputs;

                if (existingImageInputs !== savedCount) {
                    const currentCount = this._imageInputCount;

                    if (savedCount > currentCount) {
                        for (let i = currentCount + 1; i <= savedCount; i++) {
                            this.addInput(`image_${i}`, "IMAGE");
                        }
                        this._imageInputCount = savedCount;
                    } else if (savedCount < currentCount) {
                        for (let i = currentCount; i > savedCount; i--) {
                            const inputName = `image_${i}`;
                            const inputIndex = this.inputs?.findIndex(inp => inp.name === inputName);
                            if (inputIndex !== undefined && inputIndex >= 0) {
                                this.removeInput(inputIndex);
                            }
                        }
                        this._imageInputCount = savedCount;
                    }
                }
            };
        }
    },
});
