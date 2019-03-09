'use strict'
class MLP {
    constructor(options) {
        let architecture2arr, i, j, k, len,
            prevLayerSize, prevLayerNeurons
        if (typeof options === "undefined") {
            throw new Error("Uncorrect declaration of the Network."
                + " To create a neuronal network you have to inside at least the property "
                + "'architecture' that has string type value and contains layers joined with '-' "
                + "symbol. For example 'architecture' : '2-10-7-4-1' means that we have defined "
                + "a neural network with input layer of 2 neurons, output layer of 1 neuron and 3 hidden layers with"
                + " 10, 7 and 4 neurons.")
        }
        if (typeof options.architecture === "string") {
            this.architecture = options.architecture
            architecture2arr = options.architecture.split('-')
            len = architecture2arr.length
            this.inputSize = Number(architecture2arr[0])
            this.outputSize = Number(architecture2arr[len - 1])
            this.hiddenLayersSize = []
            for (i = 1; i < len - 1; i++) {
                this.hiddenLayersSize[i - 1] = Number(architecture2arr[i])
            }
        } else {
            if (typeof options.architecture === "undefined") {
                if (!Number.isInteger(options.inputSize)
                    || (options.hiddenLayersSize.constructor !== Array)
                    || !Number.isInteger(options.outputSize)) {
                    if (Number.isInteger(options.inputSize)) this.inputSize = options.inputSize
                    else this.inputSize = null
                    if ((options.hiddenLayersSize.constructor === Array)) {
                        if (options.hiddenLayersSize.length === 0) this.hiddenLayersSize = []
                        else if (options.hiddenLayersSize.every(elem => {
                            return Number.isInteger(elem)
                        })) this.hiddenLayersSize = options.hiddenLayersSize
                    } else this.hiddenLayersSize = []
                    if (Number.isInteger(this.outputSize)) this.outputSize = options.outputSize
                    else this.outputSize = null
                } else {
                    this.inputSize = options.inputSize
                    this.hiddenLayersSize = options.hiddenLayersSize
                    this.outputSize = options.outputSize
                }
            }
        }
        this.connections = (isCorrect(options.connections) === 1) ? options.connections : 'feed forward'
        this.otherParameters = {}
        if (typeof options.data === 'undefined') this.otherParameters.data = null
        else if (_isCorrectData(options.data)) this.otherParameters.data = options.data
        // create the input, hidden layers and the output
        this.otherParameters.Layers = []
        this.otherParameters.Layers[0] = new Layer({
            layer: 0,
            neurons: [],
            type: "input"
        })
        for (i = 0; i < this.inputSize; i++) {
            // complete the input layer
            // the input neurons have not
            // weights
            this.otherParameters
                .Layers[0]
                .neurons[i] = new Neuron({
                    type: "input",
                    layer: 0,
                    neuron: i,
                    output: this.otherParameters.data === null
                        ? 0.0 : this.otherParameters.data.input[0][i]
                })
        }
        // complete the hidden layer
        // the hidden layers consist neurons with
        // layer, neuron, value, inputs
        // and bias...
        for (i = 0; i < this.hiddenLayersSize.length; i++) {
            this.otherParameters.Layers[i + 1] = new Layer({
                type: 'hidden',
                layer: this._createLayerId(i + 1),
                neurons: []
            })
            for (j = 0; j < this.hiddenLayersSize[i]; j++) {
                // complete the layer with neurons
                this.otherParameters.Layers[i + 1].neurons[j] = new Neuron({
                    type: 'hidden',
                    layer: this._getLayerId(i + 1),
                    neuron: this._createNeuronId(i + 1, j),
                    inputs: [],
                    bias: 2 * random(i + j + 1, 12345) - 1,
                    output: null
                })
                // complete the inputs array:
                prevLayerSize = this.otherParameters.Layers[i].neurons.length
                prevLayerNeurons = this.otherParameters.Layers[i].neurons
                for (k = 0; k < prevLayerSize; k++) {
                    this.otherParameters.Layers[i + 1].neurons[j].inputs[k] = {
                        layer: this._getLayerId(i),
                        neuron: this._getNeuronId(i, k),
                        weight: 2 * random(i + k + 1, 12345) - 1,
                        value: prevLayerNeurons[k].output
                    }
                }
                // compute the initial activation/output
                let neuronj = this.otherParameters
                    .Layers[i + 1]
                    .neurons[j]
                neuronj.output = neuronj.activation.f(...[neuronj
                    .inputs.map(input => {
                        // out(i + 1, j) = f(Σw(j, k_i)*input.value[k_i])
                        return input.value * input.weight
                    })
                    .reduce((val1, val2) => val1 + val2, 0) + neuronj.bias, ...neuronj.activation.parameters])
            }
        }
        // complete the output layer
        // that contains neurons with 
        // layer, neuron, type, output 
        // and !!!target!!!
        let layerLen = this.otherParameters.Layers.length
        this.otherParameters.Layers.push(new Layer({
            type: "output",
            layer: this._createLayerId(layerLen),
            neurons: []
        }))
        for (i = 0; i < this.outputSize; i++) {
            this.otherParameters
                .Layers[layerLen]
                .neurons[i] = new Neuron({
                    type: "output",
                    layer: this._getLayerId(layerLen),
                    neuron: this._createNeuronId(layerLen, i),
                    bias: 2 * random(i + layerLen + 1, 12345) - 1,
                    inputs: [],
                    output: null,
                    target: this.otherParameters.data === null
                        ? 0.0 : this.otherParameters.data.output[0][i]
                })
            // complete the inputs
            prevLayerSize = this.otherParameters.Layers[layerLen - 1].neurons.length
            prevLayerNeurons = this.otherParameters.Layers[layerLen - 1].neurons
            for (j = 0; j < prevLayerSize; j++) {
                this.otherParameters
                    .Layers[layerLen]
                    .neurons[i]
                    .inputs[j] = {
                        layer: this._getLayerId(layerLen - 1),
                        neuron: this._getNeuronId(layerLen - 1, j),
                        weight: 2 * random(1 + i + j, 12345) - 1,
                        value: prevLayerNeurons[j].output
                    }
            }
            // computhe the output:
            let outputNeuroni = this.otherParameters
                .Layers[layerLen].neurons[i]
            outputNeuroni.output = outputNeuroni.activation.f(...[
                outputNeuroni.inputs.map(input => {
                    return input.weight * input.value
                })
                    .reduce((val1, val2) => val1 + val2, 0) + outputNeuroni.bias,
                ...outputNeuroni.activation.parameters
            ])
        }
    }
    data(data) {
        let inputProperties = [], outputProperties = [],
            temp, i, j, input = [], output = []
        /**
         * if the data.input/output
         *  is array then 
         * check the dimensions
         */
        if ([data, data.input, data.output].some(el => {
            return typeof el === "undefined"
        })) euriklis.nrerror('Undeclared data argument or uncorrect declaration of the data!')
        if (![data.input, data.output].every(el => {
            return el.constructor === Array
        })) euriklis.nrerror('Uncorrect declaration of input or output properties.')
        if (data.input.some(el => {
            return el.constructor === Array
        })) {
            if (!data.input.every(el => {
                return el.constructor === Array && el.length === this.inputSize
            })) euriklis.nrerror("The input data is unapropriate for this neural network.")
        }
        if (data.output.some(el => {
            return el.constructor === Array
        })) {
            if (!data.output.every(el => {
                return el.constructor === Array && el.length === this.outputSize
            })) euriklis.nrerror("The input data is unapropriate for this neural network.")
        }
        if (data.input.some(el => {
            return el.constructor === Object
        })) {
            if (!data.input.every(el => {
                return el.constructor === Object
            })) euriklis.nrerror("The input data is unapropriate for this neural network.")
            for (i = 0; i < data.input.length; i++) {
                temp = Object.keys(data.input[i])
                for (j = 0; j < temp.length; j++) {
                    if (inputProperties.indexOf(temp[j]) === -1) inputProperties.push(temp[j])
                }
            }
            if (inputProperties.length !== this.input) euriklis.nrerror("The input data is unapropriate for this neural network.")
        }
        if (data.output.some(el => {
            return el.constructor === Object
        })) {
            if (!data.output.every(el => {
                return el.constructor === Object
            })) euriklis.nrerror("The input data is unapropriate for this neural network.")
            for (i = 0; i < data.output.length; i++) {
                temp = Object.keys(data.output[i])
                for (j = 0; j < temp.length; j++) {
                    if (outputProperties.indexOf(temp[j]) === -1) outputProperties.push(temp[j])
                }
            }
            if (outputProperties.length !== this.output) euriklis.nrerror("The input data is unapropriate for this neural network.")
        }
        // re-organize the data 
        // to be only array types
        if (inputProperties.length) {
            for (i = 0; i < data.input.length; i++) {
                input[i] = []
                for (j = 0; j < inputProperties.length; j++) {
                    input[i][j] = typeof data.input[i][inputProperties[j]] === "undefined" ?
                        0.0 : data.input[i][inputProperties[j]]
                }
            }
        } else input = data.input
        if (outputProperties.length) {
            for (i = 0; i < data.output.length; i++) {
                output[i] = []
                for (j = 0; j < outputProperties.length; j++) {
                    output[i][j] = typeof data.output[i][outputProperties[j]] === "undefined" ?
                        0.0 : data.output[i][outputProperties[j]]
                }
            }
        } else output = data.output
        this.otherParameters.data = {
            input, output, inputLabels: inputProperties,
            outputLabels: outputProperties
        }
        this.activate(this.otherParameters.data.input[0])
        return this
    }
    // todo: activation method for neuron and layer!!!
    activate(example) {
        // the example has to be homogenious
        // with the data structure
        if (!this.otherParameters.data) {
            if (!this.otherParameters.messages) this.otherParameters.messages = []
            this.otherParameters.messages.push("To activate the network correctly you first have to inside data. If your example is Object the activation do not works.")
        }
        if (example.constructor === Array) {
            if (example.length !== this.inputSize) {
                throw new Error(`The example ${example} is not consistent with the data.`)
            }
            let currLayer
            for (currLayer = 0; currLayer < this.otherParameters.Layers.length; currLayer++) {
                if (currLayer === 0) {
                    let currInputNeuron
                    for (currInputNeuron = 0; currInputNeuron < this.otherParameters.Layers[currLayer].neurons.length; currInputNeuron++) {
                        this.otherParameters.Layers[currLayer]
                            .neurons[currInputNeuron].output = example[currInputNeuron]
                    }
                } else {
                    let currLayerm1 = currLayer - 1
                    for (let li = 0; li < this.hiddenLayersSize[currLayerm1]; li++) {
                        for (let ni = 0; ni < this.otherParameters.Layers[currLayerm1].neurons.length; ni++) {
                            this.otherParameters
                                .Layers[currLayer]
                                .neurons[li]
                                .inputs[ni].value = this.otherParameters.Layers[currLayerm1].neurons[ni].output
                        }
                    }
                    this.otherParameters.Layers[currLayer].activate()
                }
            }
        }
        if (example.constructor === Object) {
            if (Object.keys(example).every(prop => {
                return this.otherParameters.data['inputLabels'].some(item => {
                    return item === prop
                })
            })) {
                let tempExample = Object.values(example), currLayer
                for (currLayer = 0; currLayer < this.otherParameters.Layers.length; currLayer++) {
                    if (currLayer === 0) {
                        for (let currInputNeuron = 0; currInputNeuron < tempExample.length; currInputNeuron++) {
                            this.otherParameters
                                .Layers[currLayer]
                                .neurons[currInputNeuron]
                                .output = tempExample[currInputNeuron]
                        }
                    } else {
                        currLayerm1 = currLayer - 1
                        for (let li = 0; li < this.hiddenLayersSize[currLayerm1]; li++) {
                            for (let ni = 0; ni < this.otherParameters.Layers[currLayerm1].neurons.length; ni++) {
                                this.otherParameters
                                    .Layers[currLayer]
                                    .neurons[li]
                                    .inputs[ni] = this.otherParameters.Layers[currLayerm1].neurons[ni].output
                            }
                        }
                        this.otherParameters.Layers[currLayer].activate()
                    }
                }
            } else throw new Error(`The properties of the data are not consistent with the properties of the example.`)
        } else if (!(example instanceof Array)) throw new Error(`The example have to be Object or Array type.`)
        return this
    }
    propagate(example, check) {
        // we need the output
        // so here example is 
        // assumed to be the 
        // output from the data
        // or the output for the
        // last inserted example
        // ...................
        // if do not exists data 
        // propagation can not be done.
        if (this.otherParameters.data === null) {
            if (!this.otherParameters.messages) this.otherParameters.messages = []
            this.otherParameters.messages.push(`To make propagation is better to insert data for learning.`)
        }
        // ...................
        if (example.constructor === Array) {
            // paropagate for array
            if (example.length !== this.outputSize) {
                throw new Error(`Bad output example. The output of the network has dimension ${this.outputSize} and the example ${example.length}.`)
            }
            // check if the elements of
            // the example are numbers
            if (typeof check !== 'undefined' && check) {
                if (example.some(el => {
                    return isNaN(el)
                })) throw new Error(`The output example in propagate is NaN.`)
            }
            // L --> the last layer
            let L = this.hiddenLayersSize.length + 1,
                outNeurons = this.outputSize, layer, i
            for (i = 0; i < outNeurons; i++) this.otherParameters.Layers[L].neurons[i].target = example[i]
            // compute the error from the target
            this.otherParameters
                .Layers[L]
                .neurons.forEach(neuron => {
                    return neuron.error = neuron.target - neuron.output
                })
            this.otherParameters.Layers[L].squareError = this.otherParameters
                .Layers[L].neurons.map(el => {
                    return el.error * el.error
                }).reduce((a, b) => { return a + b }, 0)
            // compute the delta for the last layer:
            // with the formula delta[i] = f'(v[i])*error[i]
            this.otherParameters
                .Layers[L]
                .neurons
                .forEach(neuron => {
                    return neuron.delta = neuron.activation.derivative(...[
                        neuron['functional signal'],
                        ...neuron.activation.parameters
                    ]) * neuron.error
                })
            // compute the delta for the other
            // neurons with the formula:
            // Layers[l].delta[i] = f'(v[i]) * Σ w[k][i] * Layers[l + 1].delta[k]
            // where k is the number of all neurons
            // of the hidden layer l + 1
            for (layer = L - 1; layer > 0; layer--) {
                let lp1n = this.otherParameters
                    .Layers[layer + 1].neurons
                this.otherParameters
                    .Layers[layer]
                    .neurons.forEach((neuron, j) => {
                        let sumWkjTdeltak = 0
                        lp1n.forEach((neuronlp1, k) => {
                            sumWkjTdeltak += neuronlp1.delta * this._getWeight({
                                source: { neuron: j, layer: layer },
                                target: { neuron: k, layer: layer + 1 }
                            })
                        })
                        neuron.delta = neuron
                            .activation
                            .derivative(...[neuron['functional signal'],
                            ...neuron.activation.parameters
                            ]) * sumWkjTdeltak
                    })
            }
            return this
        } else {
            if (example.constructor === Object) {
                // transform the Object to array
                // and call again with the transformed
                // array...
                let transformedExample
                // the example output have to
                // be with the same property structure
                // as the data outputLabel elements
                if (Object.keys(example).length !== this.outputSize) {
                    throw new Error(`Bad output example. The example have to be with the same dimension like the output layer.`)
                }
                if (Object.values(example).some(value => {
                    return isNaN(value)
                })) throw new Error(`Bad output example.The values of the output have to be numbers.`)
                if (this.otherParameters.data === null) throw new Error(`To make propagation you firs have to insert data to the network.`)
                if (!Object.keys(example).every(function (key) {
                    return this.otherParameters.data.outputLabels.some(label => {
                        return label === key
                    })
                })) throw new Error(`All the properties of the example have to be same with the data output properties.`)
                transformedExample = Array
                    .from({ length: this.otherParameters.data.outputLabels.length })
                    .map((d, i) => {
                        let key = this.otherParameters.data.outputLabels[i]
                        return d = example[key]
                    })
                return this.propagate(transformedExample)
            } else throw new Error(`The output example is not correct. The type of an example can be Array ot Object.`)
        }
    }
    train(options) {
        // if options do not exist
        // or are uncomplete defined
        // set them to the default
        let epoch, err = 0, ofile
        if (typeof options === 'undefined') options = defaultTrainOptions('all')
        if (options instanceof Object) {
            let opt = Object.keys(options)
            if (isEmptyArray(opt)) options = defaultTrainOptions('all')
            else options = setOmitedTrainOptions(options)
            let momentum = options.momentum,
                maxiter = options['max iterations'],
                ita = options['learning rate'],
                minerror = options['minimal error'],
                wnm1 = [], wnm2 = [], l, j, i, wlji,
                inputExamples = this.otherParameters
                    .data.input, deltaj, ylm1i,
                outputExamples = this.otherParameters
                    .data.output, rndInputExamples,
                rndOutputExamples, rndExamples, biasnm1 = [], biasnm2 = [],
                inputExample, outputExample, dwnm1, bias, dbiasnm1
            // update weights:
            // .................................
            // w(l, j, i, n + 1) = w(l, j, i, n) + 
            // momentum * dw(l, j, i, n - 1) + 
            // delta(l, j, n) * y(l - 1, i, n)
            // .................................
            // where l is current layer, j is
            // the target neuron, i is the source
            // neuron (in layer l - 1), n is the
            // time period, y is the output of
            // the neuron and w is the weight
            // for i = -1 (for zero indexed)
            // arrays w is the bias of neuron j
            // and y is equals to 1. For l = 0
            // we have the input parameters for y.
            epoch = 0
            /*ofile = fs.createWriteStream('MyNetwork.docx', 'utf-8')
            ofile.write(`The architecture of the network is ${this.architecture}\n`)
            ofile.write(`The activation function is 1 / (1 + a * Math.exp(-b * x)), where a = b = 1\n`)
            ofile.write(`The derivative activation function is a * b * Math.exp(b * x) / ((a + Math.exp(b * x)) * (a + Math.exp(b * x))), where a = b = 1\n`)
            ofile.write(`The weights of the network are:\n`)
            for (l = 1;l < 2 + this.hiddenLayersSize.length;l++) {
                for (j = 0;j < this.otherParameters.Layers[l].neurons.length;j++) {
                    ofile.write(`Bias(${l}, ${j + 1}) = ${this._getBias({layer : l, neuron : j})}\n`)
                    for (i = 0;i < this.otherParameters.Layers[l - 1].neurons.length;i++) {
                        ofile.write(`W(${l}, ${j + 1}, ${i + 1}) = ${this._getWeight({
                            source : {neuron : i, layer : l - 1},
                            target : {neuron : j, layer : l}
                        })}\n`)
                    }
                }
            }*/
            do {
                ++epoch
                //ofile.write(`Epoch ${epoch}\n`)
                rndExamples = randomize(inputExamples, outputExamples)
                rndInputExamples = rndExamples.input
                rndOutputExamples = rndExamples.output
                for (let n = 0; n < rndInputExamples.length; n++) {
                    inputExample = rndInputExamples[n]
                    outputExample = rndOutputExamples[n]
                    //ofile.write(`Input example: ${JSON.stringify(inputExample)}\n`)
                    //ofile.write(`Output example: ${JSON.stringify(outputExample)}\n`)
                    this.activate(inputExample)
                        .propagate(outputExample)
                    /*for (let p = 0;p < this.outputSize;p++) {
                        ofile.write(`The error of output neuron ${p + 1} is ` 
                        + `${this.otherParameters.Layers[this.hiddenLayersSize.length + 1].neurons[p].error}\n` 
                        + ` for the target ${this.otherParameters.Layers[this.hiddenLayersSize.length + 1].neurons[p].target}\n`)
                    }*/
                    // set the weights:
                    for (l = 1; l < this.hiddenLayersSize.length + 2; l++) {
                        if (n === 0 && epoch === 1) {
                            wnm1[l] = []
                            wnm2[l] = []
                            biasnm1[l] = []
                            biasnm2[l] = []
                        }
                        for (j = 0; j < this.otherParameters.Layers[l].neurons.length; j++) {
                            if (n === 0 && epoch === 1) {
                                wnm1[l][j] = []
                                wnm2[l][j] = []
                            }
                            deltaj = this.otherParameters
                                .Layers[l].neurons[j].delta
                            /////
                            bias = this._getBias({ neuron: j, layer: l })
                            //ofile.write(`The bias of neuron ${j + 1} in layer ${l} = ${bias}\n`)
                            //ofile.write(`δ(${l}, ${j + 1}) = ${deltaj}\n`)
                            if (n < 2 && epoch === 1) dbiasnm1 = 0
                            else {
                                dbiasnm1 = biasnm1[l][j] - biasnm2[l][j]
                            }
                            this._setBias({
                                neuron: j,
                                layer: l,
                                value: bias + momentum * dbiasnm1 + ita * deltaj
                            })
                            //ofile.write(`The new bias(${l}, ${j + 1}) = ${bias + momentum * dbiasnm1 + ita * deltaj}\n`)
                            if ((n > 0 && epoch === 1) || epoch > 1) {
                                biasnm2[l][j] = biasnm1[l][j]
                            }
                            biasnm1[l][j] = bias
                            for (i = 0; i < this.otherParameters.Layers[l - 1].neurons.length; i++) {
                                wlji = this._getWeight({
                                    source: { neuron: i, layer: l - 1 },
                                    target: { neuron: j, layer: l }
                                })
                                //ofile.write(`W(${l}, ${j + 1}, ${i + 1}}) = ${wlji}\n`)
                                ylm1i = this.otherParameters
                                    .Layers[l - 1].neurons[i].output
                                //ofile.write(`y(${l - 1}, ${i + 1}) = ${ylm1i}\n`)
                                if ((n < 2 && epoch === 1)) dwnm1 = 0
                                else {
                                    dwnm1 = wnm1[l][j][i] - wnm2[l][j][i]
                                }
                                this._setWeight({
                                    source: { neuron: i, layer: l - 1 },
                                    target: { neuron: j, layer: l },
                                    value: wlji + momentum * dwnm1 + ita * deltaj * ylm1i
                                })
                                //ofile.write(`the new weight W(${l}, ${j + 1}, ${i + 1}) = ${wlji + momentum * dwnm1 + ita * deltaj * ylm1i}\n`)
                                if ((n > 0 && epoch === 1) || epoch > 1) {
                                    wnm2[l][j][i] = wnm1[l][j][i]
                                }
                                wnm1[l][j][i] = wlji
                            }
                        }
                    }
                    err += this.otherParameters.Layers[this.hiddenLayersSize.length + 1].squareError
                }
                err /= inputExamples.length
                //ofile.write(`error : ${err}\n`)
            } while (epoch <= maxiter && err > minerror)
        }
        //ofile.end()
        this.otherParameters.epochs = epoch
        this.otherParameters.error = err
        return this
    }
    addNeuron(parameters) {
        // if the parameters 
        // is undefined the neuron will
        // be added to the last hidden layer
        let layer, bias, output, type = 'hidden', inputs
        if (typeof parameters === "undefined") {
            layer = this._getLayerId(this.otherParameters.Layers.length - 2)
            bias = 2 * Math.random() - 1
            output = 0
        }
        if (parameters.constructor === Object) {
            if (!Number.isInteger(parameters.layer)) layer = this.otherParameters.Layers.length - 2
            else {
                if (parameters.layer === 0
                    || parameters.layer === this.otherParameters.Layers.length - 1) throw new Error("Neurons can be added only to hidden layers.")
                else {
                    layer = parameters.layer
                }
            }
            if (isNaN(parameters.bias)) bias = 2 * Math.random() - 1
            else bias = parameters.bias
            if (isNaN(parameters.output)) output = 0
            else output = parameters.output
        }
        neuron = this.otherParameters
            .Layers[layer]
            .neurons.length
        this.otherParameters
            .Layers[layer]
            .neurons
            .push(new Neuron({ type, layer, neuron, bias }))
        inputs = this._createNeuronConnections({ layer, neuron }).inputs
        output = this._createNeuronConnections({ layer, neuron }).output
        this.otherParameters
            .Layers[layer]
            .neurons[neuron]
            .inputs = input
        this.otherParameters
            .Layers[layer]
            .neurons[neuron]
            .output = output
        return this
    }
    //todo here!!!
    addLayer(options) {
        // we need layer id and 
        // count of neurons in the layer
        if (options.constructor !== Object) throw new Error("To add layer use Object with properties layer and neuron.")
        if (typeof options.layer === 'undefined' || typeof options.neuron === 'undefined') throw new Error(`The object do not contain property layer or neuron`)
        let layer = options.layer,
            neurons = options.neurons,
            type = options.type, add = false
        // if the layer do not exists
        // add it. This case is possible
        // if and only if we have create
        // network but have not defined
        // the architecture or the inputSize,
        // the hiddenLayersSize or outputSize
        // are not defined...
        if (typeof this.architecture === 'undefined') {
            if ((this.inputSize === null
                || this.hiddenLayersSize.constructor === Array)
                && this.outputSize === null) add = true
        }
        if (!add) throw new Error(`The network is yet created so you do not add any layer.`)
        if (this.inputSize !== null) {
            if (type === 'input') {
                if (this.otherParameters.data !== null) {
                    if (neurons !== this.otherParameters.data.input[0].length) {
                        throw new Error(`The data has dimension ${this.otherParameters.data.input[0].length} and is not consistent with the input neurons count of ${neurons}.`)
                    } else this.inputSize = neurons
                } else this.inputSize = neurons
            }
        } else if (type === 'input') this.inputSize = neurons
        if (type === 'hidden') {
            if (this.hiddenLayersSize.constructor === Array) {
                if (this.hiddenLayersSize.length > layer) {
                    throw new Error(`Do not exists layers before the layer id ${layer}.`)
                } else {
                    this.hiddenLayersSize = [
                        ...this.hiddenLayersSize.slice(0, layer),
                        neurons,
                        ...this.hiddenLayersSize.slice(layer)
                    ]
                }
            } else throw new Error(`Bad hidden layers size declaration.`)
        }
        if (type === 'output') {
            if (this.otherParameters.data === null) this.outputSize = neurons
            else {
                if (this.otherParameters.data.output[0].length === neurons) {
                    this.outputSize = neurons
                } else throw new Error(`The output neurons count is not consistent with the dimension of the output data. Change the dimension of the data and then declare the new output layer.`)
            }
        }
        let otherParamsData = { data: this.otherParameters.data !== null ? this.otherParameters.data : null },
            newNetwork = new MLP({
                inputSize: this.inputSize,
                hiddenLayersSize: this.hiddenLayersSize,
                outputSize: this.outputSize,
                otherParameters: otherParamsData
            })
        this.otherParameters = newNetwork.otherParameters
        return this
    }
    _createNeuronConnections(parameters) {
        let inputs, output
        if (!parameters.constructor === Object || !Number.isInteger(parameters.layer) || Number.isInteger(parameters.neuron)) {
            throw new Error("Internal error in createNeuronConnections.")
        }
        let layer = parameters.layer,
            neuron = parameters.neuron, i,
            source, target = { layer, neuron },
            neurons = this.otherParameters.Layers[layer - 1].neurons.length,
            weightValue = 2 * Math.random() - 1
        for (i = 0; i < neurons; i++) {
            source = { layer: layer - 1, neuron: i }
            this.otherParameters
                .Layers[layer]
                .neurons[neuron]
                .inputs[i] = Array.from({ length: neurons })
                    .map(neuronObj => {
                        neuronObj = {
                            layer: layer - 1,
                            neuron: i,
                            weight: weightValue, //this._setWeight({ source, target, value }),
                            value: this.otherParameters.Layers[layer - 1].neurons[i].output
                        }
                        return neuronObj
                    })
        }
        this.otherParameters
            .Layers[layer]
            .neurons[neuron].output = this.otherParameters
                .Layers[layer] // the activate method used bellow
                .neurons[neuron].activate() // is for the Neuron object!!!
        inputs = this.otherParameters.Layer[layer].neurons[neuron].inputs
        output = this.otherParameters.Layer[layer].neurons[neuron].output
        return { inputs, output }
    }
    _createLayerId(layer) {
        let llen, msg = ``, tempData
        // if this layer exists
        // then re-slice the array
        // of the layers and update the 
        // array of the layer id's
        //1.Get the layers length
        llen = this.otherParameters.Layers.length
        //2.if len = 0 and layer > 0
        // set layer = 0 and write
        // message.
        if (llen === 0) {
            this.addMessage(`Because there are no layers the layer ${layer} can not be added so it will be set to 0.`)
            layer = 0
        }
        // 3. If layer is greater than llen
        // do not add any layer and throw
        // error message
        if (layer > llen) {
            msg = `The length of all layers is ${llen} and ` +
                `the layer that was included is ${layer}, so this layer` +
                'do not exists. Please check your code'
            throw new Error(msg)
        } else {
            // 4. If the layer is the last
            // index of the layer set the layer
            // property of otherParameters.Layer[llen - 1]
            // to be equals to layer...
            if (layer === llen) return layer
            else {
                // 5. If the layer is smaller than llen - 1
                // then re-compute the id's of every layer
                this.hiddenLayersSize = [
                    ...this.hiddenLayersSize.slice(0, layer),
                    this.otherParameters.Layers[layer].neurons.length,
                    ...this.hiddenLayersSize.slice(layer, llen)
                ]
                let This = new MLP({
                    inputSize: this.inputSize,
                    hiddenLayersSize: this.hiddenLayersSize,
                    outputSize: this.outputSize,
                    connections: this.connections
                })
                this.inputSize = This.inputSize
                this.outputSize == This.outputSize
                this.hiddenLayersSize = This.hiddenLayersSize
                if (typeof this.otherParameters.data !== 'undefined') {
                    tempData = this, this.otherParameters.data
                }
                this.otherParameters = This.otherParameters
                this.otherParameters.data = tempData
                return layer
            }
        }
    }
    _getLayerId(lid) {
        let layers = this.otherParameters.Layers.length
        if (lid >= layers) throw new Error(`This network contains ${layers} layers including the input and output layer. So the layer with id ${lid} do not exists.`)
        else return this.otherParameters.Layers[lid].layer
    }
    _getNeuronId(lid, nid) {
        let layers = this.otherParameters.Layers.length
        if (lid >= layers) throw new Error(`The length of the layers of the network is ${layers}. The layer with id ${lid} do not exists.`)
        let neuronsOfLid = this.otherParameters.Layers[lid].neurons.length
        if (nid >= neuronsOfLid) throw new Error(`The number of the neurons of layer ${lid} is ${neuronsOfLid}. The neuron with id ${nid} do not exists.`)
        return this.otherParameters.Layers[lid].neurons[nid].neuron
    }
    _createNeuronId(lid, nid) {
        // 1. If lid is bigger than layers
        // then trow Error ...
        let layers = this.otherParameters.Layers.length
        if (lid >= layers) throw new Error(`The layers of the network are ${layers}. The layer id have to be smaller than this number, so the layer id ${lid} do not exists.`)
        // 2. if the meuron id is for new neuron create it
        if (nid === this.otherParameters.Layers[lid].neurons.length) return Number(nid)
        else throw new Error(`The neuron id ${nid} for layer id ${lid} exists.`)
    }
    addMessage(msg) {
        if (typeof this.otherParameters.messages === 'undefined') {
            this.otherParameters.messages = []
        } else {
            if (this.otherParameters.messages.constructor === Array) {
                this.otherParameters.messages.push(msg.toString())
            } else throw new Error(`Internal error: the message property of oherParameters is not Array type.`)
        }
    }
    _setWeight(parameters) {
        let targetLayer = this.otherParameters.Layers[parameters.target.layer],
            targetNeuron = targetLayer.neurons[parameters.target.neuron],
            sourceLayer = parameters.source.layer, val = parameters.value,
            sourceNeuron = parameters.source.neuron, weightIndex
        // hint : the targetNeuron structure is:
        // {..., inputs : [many objects: {layer, neuron, weight, value}]}
        weightIndex = targetNeuron.inputs.findIndex(neuron => {
            return neuron.layer === sourceLayer && neuron.neuron === sourceNeuron
        })
        if (typeof weightIndex === 'undefined') throw new Error(`Internal error: The source neuron or layer in setWeight are not correct.`)
        if (isNaN(val)) throw new Error(`The value property in setWeight is NaN.`)
        this.otherParameters
            .Layers[parameters.target.layer]
            .neurons[parameters.target.neuron]
            .inputs[weightIndex]
            .weight = val
    }
    _getWeight(parameters) {
        let targetLayer = this.otherParameters.Layers[parameters.target.layer],
            targetNeuron = targetLayer.neurons[parameters.target.neuron],
            sourceLayer = parameters.source.layer,
            sourceNeuron = parameters.source.neuron, weightIndex
        // hint : the targetNeuron structure is:
        // {..., inputs : [many objects: {layer, neuron, weight, value}]}
        weightIndex = targetNeuron.inputs.findIndex(neuron => {
            return neuron.layer === sourceLayer && neuron.neuron === sourceNeuron
        })
        if (typeof weightIndex === 'undefined') throw new Error(`Internal error: The source neuron or layer in getWeight are not correct.`)
        return targetNeuron.inputs[weightIndex].weight
    }
    _setBias(parameters) {
        let layers = this.otherParameters
            .Layers
        if (layers.length <= parameters.layer) {
            throw new Error(`The declared layer property in setBias is bigger than the dimension of the network layers.`)
        }
        let neurons = layers[parameters.layer]
            .neurons
        if (neurons.length <= parameters.neuron) {
            throw new Error(`The declared neuron property in setBias is bigger than the dimension of the layer id ${parameters.layer}.`)
        }
        if (isNaN(parameters.value)) throw new Error(`The parameter in sebBias is NaN.`)
        this.otherParameters
            .Layers[parameters.layer]
            .neurons[parameters.neuron]
            .bias = parameters.value
    }
    _getBias(parameters) {
        let layers = this.otherParameters
            .Layers
        if (layers.length <= parameters.layer) {
            throw new Error(`The declared layer property in getBias is bigger than the dimension of the network layers.`)
        }
        let neurons = layers[parameters.layer]
            .neurons
        if (neurons.length <= parameters.neuron) {
            throw new Error(`The declared neuron property in getBias is bigger than the dimension of the layer id ${parameters.layer}.`)
        }
        return neurons[parameters.neuron]
            .bias
    }
    getBias(parameters) {
        if (typeof parameters === 'undefined') {
            throw new Error(`The parameters layer and neuron for get bias are not defined.`)
        }
        if (!(parameters instanceof Object)) throw new Error(`The type of the argument in getBias have to be of Object type and it is of ${typeof parameters} type`)
        if (Object.keys(parameters).some(parameter => {
            parameter !== 'layer' || parameter !== 'neuron'
        }) || Object.keys(parameters.length !== 2)) throw new Error(`Bad declaration of the argument in getBias.`)
        return this._getBias(parameters)
    }
    setBias(parameters) {
        if (typeof parameters === 'undefined') {
            throw new Error(`The parameters layer and neuron for set bias are not defined.`)
        }
        if (!(parameters instanceof Object)) throw new Error(`The type of the argument in setBias have to be of Object type and it is of ${typeof parameters} type.`)
        if (Object.keys(parameters).some(parameter => {
            parameter !== 'layer' || parameter !== 'neuron'
        }) || Object.keys(parameters.length !== 2)) throw new Error(`Bad declaration of the argument in setBias.`)
        this._setBias(parameters)
        return this
    }
    getWeight(parameters) {
        // 1. The needed parameters are
        // source --> value of object type
        // target --> value of object type
        if (!(parameters instanceof Object)) {
            throw new Error(`The object parameter of getWeight method is not defined.`)
        }
        if (!(parameters.source instanceof Object)) {
            throw new Error(`The source property of getWeight's parameter has to be Object type.`)
        } else {
            if (parameters.source.layer instanceof Number) {
                if (parameters.source.layer > this.otherParameters.Layers.length - 2) {
                    throw new Error(`Error in getWeight.The layer id ${parameters.source.layer} can not be source.`)
                }
            } else throw new Error(`The layer property of source key in getWeight's parameter is NaN.`)
            if (parameters.source.neuron instanceof Number) {
                if (parameters
                    .source.neuron > this.otherParameters
                        .Layers[parameters.source.layer].neurons.length - 1) {
                    throw new Error(`Error in getWeight.The neuron ${parameters.source.neuron}`
                        + `can not be source neuron for layer ${parameters.source.layer}.`)
                }
            } else throw new Error(`The neuron property of source key in getWeight's parameter is NaN.`)
        }
        if (!(parameters.target instanceof Object)) {
            throw new Error(`The target property of getWeight's parameter has to be Object type.`)
        } else {
            if (parameters.target.layer instanceof Number) {
                if (parameters.target.layer > this.otherParameters.Layers.length - 1) {
                    throw new Error(`Error in getWeight.The layer id ${parameters.target.layer} can not be target.`)
                }
            } else throw new Error(`The layer property of target key in getWeight's parameter is NaN.`)
            if (parameters.target.neuron instanceof Number) {
                if (parameters
                    .target.neuron > this.otherParameters
                        .Layers[parameters.target.layer].neurons.length - 1) {
                    throw new Error(`Error in getWeight.The neuron ${parameters.target.neuron}`
                        + `can not be target neuron for layer ${parameters.target.layer}.`)
                }
            } else throw new Error(`The neuron property of target key in getWeight's parameter is NaN.`)
        }
        this._getWeight(parameters)
    }
    setWeight(parameters) {
        // 1. The needed parameters are
        // source --> value of object type
        // target --> value of object type
        // value --> value of number type
        if (!(parameters instanceof Object)) {
            throw new Error(`The object parameter of setWeight method is not defined.`)
        }
        if (!(parameters.source instanceof Object)) {
            throw new Error(`The source property of setWeight's parameter has to be Object type.`)
        } else {
            if (parameters.source.layer instanceof Number) {
                if (parameters.source.layer > this.otherParameters.Layers.length - 2) {
                    throw new Error(`Error in setWeight.The layer id ${parameters.source.layer} can not be source.`)
                }
            } else throw new Error(`The layer property of source key in setWeight's parameter is NaN.`)
            if (parameters.source.neuron instanceof Number) {
                if (parameters
                    .source.neuron > this.otherParameters
                        .Layers[parameters.source.layer].neurons.length - 1) {
                    throw new Error(`Error in setWeight.The neuron ${parameters.source.neuron}`
                        + `can not be source neuron for layer ${parameters.source.layer}.`)
                }
            } else throw new Error(`The neuron property of source key in setWeight's parameter is NaN.`)
        }
        if (!(parameters.target instanceof Object)) {
            throw new Error(`The target property of setWeight's parameter has to be Object type.`)
        } else {
            if (parameters.target.layer instanceof Number) {
                if (parameters.target.layer > this.otherParameters.Layers.length - 1) {
                    throw new Error(`Error in setWeight.The layer id ${parameters.target.layer} can not be target.`)
                }
            } else throw new Error(`The layer property of target key in setWeight's parameter is NaN.`)
            if (parameters.target.neuron instanceof Number) {
                if (parameters
                    .target.neuron > this.otherParameters
                        .Layers[parameters.target.layer].neurons.length - 1) {
                    throw new Error(`Error in setWeight.The neuron ${parameters.target.neuron}`
                        + `can not be target neuron for layer ${parameters.target.layer}.`)
                }
            } else throw new Error(`The neuron property of target key in setWeight's parameter is NaN.`)
        }
        if (typeof parameters.value === 'undefined' || isNaN(parameters.value)) {
            throw new Error(`The value property of setWeight's parameter is NaN or undefined.`)
        } else {
            // 2. set the weight:
            this._setWeight(parameters)
            return this
        }
    }
}
// todo: createNeuronConnections method ...
// todo: activation.f and the others utility functions
const isEmptyArray = (arr) => {
    return arr === "undefined" || arr.length === 0
},
    isCorrect = (parameter) => {
        if (typeof parameter === "string") {
            if (['feed forward', 'Hopfield', 'SOM', 'PCA', 'RBF', 'SVM'].some(connection => {
                return connection === parameter
            })) return 1
        }
        if (typeof parameter === "undefined") return 2
        return 0
    }
function randomize(inputExamples, outputExamples) {
    let arra1 = Array
        .from({ length: inputExamples.length })
        .map((el, i) => { return el = inputExamples[i] }),
        arra2 = Array
            .from({ length: outputExamples.length })
            .map((el, i) => { return el = outputExamples[i] }),
        ctr = arra1.length,
        temp1, temp2, index,
        cplist
    // While there are elements in the array
    while (ctr > 0) {
        // Pick a random index
        index = Math.floor(Math.random() * ctr)
        // Decrease ctr by 1
        ctr--
        // And swap the last element with it
        temp1 = arra1[ctr]
        temp2 = arra2[ctr]
        arra1[ctr] = arra1[index]
        arra2[ctr] = arra2[index]
        arra1[index] = temp1
        arra2[index] = temp2
    }
    cplist = {}
    cplist.input = arra1
    cplist.output = arra2
    return cplist
}
function defaultTrainOptions(option) {
    if (typeof option === 'undefined') option = 'all'
    if (typeof option !== 'string') throw new Error(`Internal error in default train options.`)
    let options = {
        'learning rate': 0.05,
        'momentum': 0.0,
        'max iterations': 6000,
        'minimal error': 1e-6,
        'adaptive control': 'constant learning rate'
    }
    if (option === 'all') return options
}
function setOmitedTrainOptions(options) {
    let allOpts = defaultTrainOptions('all'),
        allProp = Object.keys(allOpts)
    allProp.forEach(prop => {
        if (options[prop] === null || typeof options[prop] === 'undefined') {
            options[prop] = allOpts[prop]
        }
    })
    return options
}
function random(m,seed){
    var i,j,k,rn
    for(i = 0;i < m;i++){
            seed = parseInt(seed);
            k = parseInt(seed/127773);
            seed = parseInt(16807*(seed - k*127773)- k*2836);
            if ( seed < 0 )seed += 2147483647;
            rn = seed*4.656612875e-10;
    }
    return rn;
}
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
        f: availableFunctions['logistic'].f,
        derivative: availableFunctions['logistic'].derivative,
        parameters: availableFunctions['logistic'].defaultParameters
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
            })
                .reduce((val1, val2) => { return val1 + val2 }, this.bias)
            this.output = this.activation.f(...[
                this['functional signal'],
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
// test for XOR problem:
let net = new MLP({architecture : '2-2-1'})
.data({input : [[1,1], [0, 0], [1, 0], [0, 1]], output : [[0], [0], [1], [1]]})
.train({'max iterations' : 20000})
console.log(net)
/*
   Bad output:
{
  "architecture": "2-2-1",
  "inputSize": 2,
  "outputSize": 1,
  "hiddenLayersSize": [
    2
  ],
  "connections": "feed forward",
  "otherParameters": {
    "data": {
      "input": [
        [1,1],
        [ 0,0],
        [1,0],
        [0,1]
      ],
      "output": [[0],[0],[1],[1]],
      "inputLabels": [],
      "outputLabels": []
    },
    "Layers": [
      {
        "type": "input",
        "layer": 0,
        "neurons": [
          {
            "layer": 0,
            "neuron": 0,
            "type": "input",
            "output": 1
          },
          {
            "layer": 0,
            "neuron": 1,
            "type": "input",
            "output": 1
          }
        ]
      },
      {
        "type": "hidden",
        "layer": 1,
        "neurons": [
          {
            "layer": 1,
            "neuron": 0,
            "type": "hidden",
            "inputs": [
              {
                "layer": 0,
                "neuron": 0,
                "weight": 12.486679112679981,
                "value": 1
              },
              {
                "layer": 0,
                "neuron": 1,
                "weight": 12.487694382307714,
                "value": 1
              }
            ],
            "activation": {
              "f": /**id:1d**/ (x, a, b) => { return 1 / (1 + a * Math.exp(-b * x)) },
              "derivative": /**id:1e**/ (x, a, b) => { return a * b * Math.exp(b * x) / ((a + Math.exp(b * x)) * (a + Math.exp(b * x))) },
              "parameters": [
                /**id:1f**/
                1,
                1
              ]
            },
            "bias": -5.7323237470412645,
            "output": 0.9999999956017089,
            "functional signal": 19.24204975275848,
            "delta": -3.208032517433969e-8
          },
          {
            "layer": 1,
            "neuron": 1,
            "type": "hidden",
            "inputs": [
              {
                "layer": 0,
                "neuron": 0,
                "weight": -12.615675394876433,
                "value": 1
              },
              {
                "layer": 0,
                "neuron": 1,
                "weight": 13.02454193664693,
                "value": 1
              }
            ],
            "activation": {
              "f": /**ref:1d**/,
              "derivative": /**ref:1e**/,
              "parameters": /**ref:1f**/
            },
            "bias": 6.400731578574613,
            "output": 0.9988968641332844,
            "functional signal": 6.808494622275047,
            "delta": 0.007356653800420436
          }
        ]
      },
      {
        "type": "output",
        "layer": 2,
        "neurons": [
          {
            "layer": 2,
            "neuron": 0,
            "type": "output",
            "inputs": [
              {
                "layer": 1,
                "neuron": 0,
                "weight": 61.265218045267645,
                "value": 0.30857986811774707
              },
              {
                "layer": 1,
                "neuron": 1,
                "weight": -56.08904980788667,
                "value": 0.6610527735810995
              }
            ],
            "activation": {
              "f": /**ref:1d**/,
              "derivative": /**ref:1e**/,
              "parameters": /**ref:1f**/
            },
            "bias": 18.069473448210573,
            "output": 0.47716177185242736,
            "target": 0,
            "error": -0.47716177185242736,
            "functional signal": -0.0914165232750932,
            "delta": -0.11904156270022229
          }
        ],
        "squareError": 0.22768335651734795
      }
    ],
    "epochs": 20001,
    "error": 0.335993992401844
  }
}
*/
