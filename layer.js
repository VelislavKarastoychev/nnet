'use strict'
class Layer {
    constructor(options) {
        if (typeof options === "undefined") {
            throw new Error("Uncorrect Layer declaration. To create an layer you have to inside an object with properties 'layer' and 'neurons'.")
        }
        this.type = options.type || 'hidden'
        this.layer = options.layer
        this.neurons = typeof options.neurons === "undefined"
            ? [] : options.neurons.constructor === Array
                ? options.neurons : []
        if (Number.isInteger(options.layer)) {
            this.layer = options.layer
        }
    }
    activate() {
        let neurons = this.neurons.length,
            neuron
        for (neuron = 0; neuron < neurons; neuron++) {
            this.neurons[neuron].activate()
        }
    }
}
module.exports.Layer = Layer
