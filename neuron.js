'use strict'
function createLayerID(lid) {
    if (!Number.isInteger(lid)) {
        throw new Error(`The layer id have to be number type`)
    } else return Number(lid)
}
function createId(nid) {
    if (!Number.isInteger(nid)) throw new Error(`The neuron id have to be only number type.`)
    else return Number(nid)
}
let availableFunctions = {
    "tanh": {
        f: (x, a, b) => { return a * Math.tanh(b * x) },
        derivative: (x, a, b) => { return a * b * (1 - Math.tanh(b * x) * Math.tanh(b * x)) },
        defaultParameters: [1.7159, 2 / 3]
    },
    "logistic": {
        f: (x, a, b) => { return 1 / (1 + a * Math.exp(-b * x)) },
        derivative: (x, a, b) => { return a * b * Math.exp(b * x) / ((a + Math.exp(b * x)) * (a + Math.exp(b * x))) },
        defaultParameters: [1, 1]
    },
    "ReLU": {
        f: (x, a, b) => {
            if ((a * x + b) < 0) return 0
            else return x
        },
        derivative: (x, a, b) => {
            if ((a * x + b) < 0) return 0
            else return 1
        },
        defaultParameters: [1, 0]
    }
}
/*   availableAdaptiveRateControl = {
        "Darken and Moody": {
            f: (t, ita0, tau) => {
                return ita0 * tau / (t + tau)
            },
            "Initial ita": 0.1,
            "Tau": Math.max(...[this.otherParameters.data.input.length / 10 | 0, 1])
        },
        "Haykin": {
            f: (n, nswitch, alpha) => {
                return (nswitch / (n + nswitch)) * (alpha / this.getWeightsIverseHessian().eigenvalueByPowerMethod())
            },
            'Initial n switch': 5,
            'Alpha': 1
        },
        'Murata': {
            f: (r, ita, alpha, beta, gamma) => {
                if (typeof r === 'undefined') r = [Array.from({ length: this.output }).map(el => { return el = 0; })].toMatrix()
                if (typeof ita === 'undefined') ita = 0.1
                r = r.plus(this.otherParameters.training.y.times(gamma))
                ita = ita + alpha * ita * (beta * r.FrobeniusNorm() - ita)
                return { ita, r, alpha, beta, gamma }
            },
            'Initial ita': 0.1,
            'Alpha': 0.5, 'Beta': 0.5, 'Gamma': 0.4
        }
    } */
function createActivation(options) {
    // 1. options are not defined 
    // set to the default activation 
    // function...
    if (typeof options === 'undefined') return {
        f: availableFunctions['tanh'].f,
        derivative: availableFunctions['tanh'].derivative,
        parameters: availableFunctions['tanh'].defaultParameters
    }
    // 2. If options are string check
    // if the function is available
    // and set the activation to the
    // available function...
    if (typeof options === 'string') {
        if (availableFunctions.some(functionObj => {
            return functionObj.f === options.toLowerCase()
        })) {
            let strOpt = options.toLowerCase(),
                optFunction = availableFunctions[strOpt]
            return {
                f: optFunction.f,
                derivative: optFunction.derivative,
                parameters: optFunction.defaultParameters
            }
        }
    }
    if (typeof options === 'object') {
        // 3. If the options is Object type
        // then check if the f property is 
        // string that is equal to the 
        // parameters of the available functions,
        // set the f property to the
        // available function and set the
        // parameters property to the 
        // parameters key of the options
        // if options.parameters do not 
        // exists set the parameters to 
        // the default parameters of the
        // available function
        if (typeof options.f === 'string') {
            if (availableFunctions.some(functionObj => {
                return functionObj.f === options.f.toLowerCase()
            })) {
                let strOpt = options.f.toLowerCase(),
                    optFunction = availableFunctions[strOpt],
                    currParameters
                if (options.parameters.constructor === Array) {
                    if (options.parameters.every(param => {
                        return !isNaN(param)
                            && options
                                .parameters
                                .length === optFunction
                                    .defaultParameters
                                .length
                    })) currParameters = options.parameters
                    else currParameters = optFunction.defaultParameters
                } else currParameters = optFunction.defaultParameters
                return {
                    f: optFunction.f,
                    derivative: optFunction.derivative,
                    parameters: currParameters
                }
            } else throw new Error(`The activation function ${options.f} is not available.`)
        }
        if (options.f instanceof Function
            && options.derivative instanceof Function) {
            if (options.parameters.constructor === Array) {
                if (options.parameters.length === options.f.arguments - 1) {
                    return options
                } else throw new Error(`The parameters argument is not correct.`)
            }
        } else throw new Error(`The options in activate method are not correctly declared!`)
    } else {
        options = null
        return createActivation(options)
    }
}
class Neuron {
    constructor(parameters) {
        if (typeof parameters !== "undefined") {
            if (typeof parameters.layer !== 'undefined') {
                this.layer = createLayerID(parameters.layer)
            } else throw new Error("Uncorrect layer id parameter declaration in Neuron.")
            if (typeof parameters.neuron !== 'undefined') {
                this.neuron = createId(parameters.neuron)
            } else throw new Error("Uncorrect id parameter declaration in Neuron.")
            if (typeof parameters.inputs !== "undefined") {
                let correctInputs = parameters.inputs.every(input => {
                    let isLegalInput = input.constructor === Object
                        && typeof input.layer === "number"
                        && typeof input.neuron === "number"
                        && typeof input.value === "number"
                        && typeof input.weight === "number"
                        && typeof input.bias === "number"
                    return isLegalInput
                })
                if (!correctInputs) throw new Error(`Uncorrect input parameters in neuron ${this.id} of layer ${this.layerID}`)
            }
        }
        this.type = typeof parameters.type === "undefined"
            ? "hidden" : parameters.type
        if (this.type === 'hidden' || this.type === 'output') {
            this.inputs = typeof parameters.inputs === "undefined"
                ? [] : parameters.inputs
        }
        // array with object elements
        // that have the following properties:
        // 1."layer", 2."neuron", 3. "value",
        // 4. "weight", 5.activation
        if (this.type !== 'input') this.activation = createActivation(parameters.activation)
        // object with properties:
        // 1. f --> the activation function
        // 2. derivative --> the derivative function of f
        // 3. parameters --> an array with additional
        // values for the activation function
        if (this.type !== 'input') {
            this.bias = typeof parameters.bias === 'undefined'
                ? null : parameters.bias
        }
        this.output = parameters.output || null
        if (this.type !== 'input') this.activate()
        // null type
        if (this.type === 'output') {
            this.target = null
            this.error = null
        }
    }
    activate() {
        if (this._isCompleted()) {
            this['functional signal'] = this.inputs.map(input => {
                return input.weight * input.value
            }).reduce((val1, val2) => { return val1 + val2 }, this.bias)

            this.output = this.activation.f(...[this['functional signal'],
            ...this.activation.parameters
            ])
        }
    }
    _setActivation(opt) {
        this.activation = createActivation(opt)
    }
    _isCompleted() {
        // 1.The bias have to be declared.
        // 2.The weights have to be declared.
        // 3.The values (the output of the 
        // input neurons) have to be declared.
        if (isNaN(this.bias)) return false
        if (this.inputs.length === 0) return false
        if (this.inputs.some(input => input.constructor !== Object)) {
            return false
        }
        if (this.inputs.some(input => {
            typeof input.weight === 'undefined'
                || typeof input.weight !== 'number'
                || typeof input.value === 'undefined'
                || typeof input.value !== 'number'
        })) return false
        return true
    }
}
module.exports = { Neuron }
