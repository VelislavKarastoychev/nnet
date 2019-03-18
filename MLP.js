'use strict'
let euriklis = require('euriklis')
let { Neuron } = require('./neuron')
let { Layer } = require('./layer')
let fs = require('fs')
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
    activate(example, output) {
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
                        this.otherParameters
                            .Layers[currLayer]
                            .neurons[currInputNeuron].output = example[currInputNeuron]
                    }
                    //this.otherParameters.Layers[currLayer].activate()
                } else {
                    let currLayerm1 = currLayer - 1
                    for (let li = 0; li < this.otherParameters.Layers[currLayer].neurons.length; li++) {
                        for (let ni = 0; ni < this.otherParameters.Layers[currLayerm1].neurons.length; ni++) {
                            this.otherParameters
                                .Layers[currLayer]
                                .neurons[li]
                                .inputs[ni].value = this.otherParameters
                                    .Layers[currLayerm1].neurons[ni].output
                        }
                    }
                    this.otherParameters.Layers[currLayer].activate()
                }
            }
            if (output) this.computeError(output)
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
    train(options) {
        // if options do not exist
        // or are uncomplete defined
        // set them to the default
        if (typeof options === 'undefined') options = defaultTrainOptions('all')
        if (dataDoNotExists(this)) throw new Error('To train the network inserting of data is required, so insert data by the data method.')
        if (options instanceof Object) {
            // checkForCorrectTrainingMethod(options)
            switch (options.method) {
                case 'backpropagation':
                    console.log('go to backpropagation')
                    this.backpropagation(options)
                    break
                case 'quickprop':
                    this._quickprop(options)
                    break
                case 'cgd':
                    this._conjugateGradientDescent(options)
                    break
                case 'conjugate gradient descent':
                    this._conjugateGradientDescent(options)
                    break
            }
        }
        return this
    }
    backpropagation(options) {
        if (!(options instanceof Object)) throw new Error(`Internal Error: non declared options for backpropagation`)
        // initialization of the variables:
        const neuronsInLayer = (l) => {
            return this.otherParameters
                .Layers[l].neurons.length
        }
        let epoch = 0, err = 0,
            opt = Object.keys(options)
        if (isEmptyArray(opt)) options = defaultTrainOptions('all')
        else options = setOmitedTrainOptions(options)
        let momentum = options.momentum,
            maxiter = options['max iterations'],
            ita = options['learning rate'],
            minerror = options['minimal error'],
            wnm1 = [], wnm2 = [], l, j, i, wlji,
            inputExamples = this.otherParameters
                .data.input, deltaj, ylm1i, n,
            outputExamples = this.otherParameters
                .data.output, rndInputExamples,
            rndOutputExamples, rndExamples, biasnm1 = [], biasnm2 = [],
            inputExample, outputExample, dwnm1, bias, dbiasnm1,
            L = this.otherParameters
                .Layers.length// depth of the network
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
        // ...................................
        // compute the initial error:
        inputExamples.forEach((input, i) => {
            err += this.activate(input)
                .computeError(outputExamples[i])
            err /= inputExamples.length
        })
        do {
            // step 0: Check for stop criteria:
            if (epoch > maxiter || err < minerror) break
            ++epoch
            // step 1: randomize the data
            rndExamples = randomize(inputExamples, outputExamples)
            rndInputExamples = inputExamples// rndExamples.input
            rndOutputExamples = outputExamples//rndExamples.output
            for (n = 0; n < rndInputExamples.length; n++) {
                inputExample = rndInputExamples[n]
                outputExample = rndOutputExamples[n]
                // step 2: For every example make
                // activation of the network...
                this.activate(inputExample, outputExample)
                // step 3: Compute the deltas for
                // every layer and update the weights
                // back propagation...
                for (l = L - 1; l > 0; l--) {
                    if (n === 0 && epoch === 1)
                        if (momentum !== 0) {
                            biasnm1[l] = []
                            biasnm2[l] = []
                            wnm1[l] = []
                            wnm2[l] = []
                        }
                    // compute the delta
                    for (j = 0; j < neuronsInLayer(l); j++) {
                        if (l === L - 1) {
                            this.otherParameters
                                .Layers[l]
                                .neurons[j]
                                .delta = this.otherParameters
                                    .Layers[l]
                                    .neurons[j]
                                    .activation.derivative(...[
                                        this.otherParameters
                                            .Layers[l]
                                            .neurons[j]['functional signal'],
                                        ...this.otherParameters
                                            .Layers[l]
                                            .neurons[j]
                                            .activation.parameters
                                    ]) * this.otherParameters
                                        .Layers[l]
                                        .neurons[j]
                                    .error
                        } else {
                            let sumWkjTdeltak = 0
                            for (let k = 0; k < neuronsInLayer(l + 1); k++) {
                                sumWkjTdeltak += this.otherParameters
                                    .Layers[l + 1]
                                    .neurons[k]
                                    .delta * this._getWeight({
                                        source: { layer: l, neuron: j },
                                        target: { layer: l + 1, neuron: k }
                                    })
                            }
                            this.otherParameters
                                .Layers[l]
                                .neurons[j]
                                .delta = this.otherParameters
                                    .Layers[l]
                                    .neurons[j]
                                    .activation.derivative(...[
                                        this.otherParameters.Layers[l]
                                            .neurons[j]['functional signal'],
                                        ...this.otherParameters
                                            .Layers[l]
                                            .neurons[j]
                                            .activation.parameters
                                    ]) * sumWkjTdeltak
                        }
                        // update the bias
                        let blj = this._getBias({ layer: l, neuron: j })
                        deltaj = this.otherParameters
                            .Layers[l].neurons[j].delta
                        if (momentum !== 0) {
                            if (n === 0 && epoch === 1) {
                                biasnm1[l][j] = 0.0
                                wnm1[l][j] = []
                                wnm2[l][j] = []
                            }
                            biasnm2[l][j] = biasnm1[l][j]
                            biasnm1[l][j] = blj
                        }
                        if (((n > 1 && epoch === 1) || epoch > 1) && momentum !== 0) {
                            dbiasnm1 = biasnm1[l][j] - biasnm2[l][j]
                        } else dbiasnm1 = 0
                        //console.log(`db(${l}), ${j + 1}) = ${dbiasnm1}`)
                        blj = blj + momentum * dbiasnm1 + ita * deltaj
                        this._setBias({ layer: l, neuron: j, value: blj })
                        for (i = 0; i < neuronsInLayer(l - 1); i++) {
                            // update the weights
                            let source = { layer: l - 1, neuron: i },
                                target = { layer: l, neuron: j },
                                ylm1i = this.otherParameters
                                    .Layers[l - 1]
                                    .neurons[i].output
                            if (momentum !== 0) {
                                if (n === 0 && epoch === 1) wnm1[l][j][i] = 0.0
                                wnm2[l][j][i] = wnm1[l][j][i]
                                wnm1[l][j][i] = wlji
                            }
                            if (momentum !== 0 && ((n > 1 && epoch === 1) || epoch > 1)) {
                                dwnm1 = wnm1[l][j][i] - wnm2[l][j][i]
                            } else dwnm1 = 0
                            wlji = this._getWeight({ source, target })
                            //console.log(`dw(${l}, ${j + 1}, ${i + 1}) = ${dwnm1}`)
                            wlji += momentum * dwnm1 + ita * deltaj * ylm1i
                            this._setWeight({
                                source,
                                target,
                                value: wlji
                            })
                        }
                    }
                }
            }
            err = 0
            inputExamples.forEach((input, i) => {
                err += this.activate(input)
                    .computeError(outputExamples[i])
                err /= inputExamples.length
            })
        } while (1)
        this.otherParameters.epochs = epoch
        this.otherParameters['training error'] = err
        return this
    }
    _quickprop(options) {
        // implementation of the 
        // Fahlman algorithm. The
        // adaptive formula is given
        // from Konstantinos Diamandaras
        // "Τεχνιτά Νευρωνικά δίκτυα", 
        // press "Κλειδά«ριθμος", 2007, 
        // pp. 69-70 and also in the book
        // "Neural networks and statistical learning"
        // M.N.s. Swamy and K.-L. Du, pp. 111
        // set the additionally the options:
        if (isNaN(options['max iterations'])) {
            options['max iterations'] = 10
        }
        if (isNaN(options['minimal error'])) {
            options['minimal error'] = 1e-4
        }
        if (isNaN(options['maximum growth factor'])) {
            options['maximum growth factor'] = 1.75
        }
        // check if data is declared:
        // updatind weights formula
        // Δw(l, i, j, t + 1) = 
        // Δw(l, i, j, t) 
        // * [g(l, i, j, t) / (g(l, i, j, t - 1) - g(l, i, j, t)]
        // g = J'(w)
        // Using of the gradient function
        // this function make use of two parameters
        // the function and an one - dimensional Array
        // of the initial vector argument
        const gradient = euriklis.Mathematics.gradient,
            minerr = options['minimal error'],
            maxiterations = options['max iterations'],
            cost =  (w) => {
                // w is an one - dimension array
                // that composite the weights in 
                // an arbitrary order
                setNetworkWeights(w, this)
                let err = this.activate(example.input)
                    .computeError(example.output)
                this.otherParameters.Layers[this.hiddenLayersSize.length + 1]
                    .squareError = this.otherParameters.Layers[this.hiddenLayersSize.length + 1]
                        .neurons.map(neuron => {
                            return neuron.error * neuron.error
                        }).reduce(sumarize, 0)
                return err
            },
            examples = areInsertedExampelesInOptions(options) ?
                options.examples : this.otherParameters.data
        let w0 = orderWeightsTo1DimArray(this), it = 0,
            err = (example) => {
                return this.activate(example.input)
                    .computeError(example.output)
            }, g, dw = Array.from({ length: w0.length })
                .map(wi => {
                    return wi = 0
                }),
            gm1 = Array({ length: w0.length })
                .map(gi => {
                    return gi = 0
                }), mju = options['maximum growth factor'],
            ita = options['learning rate'], example, k,
            gerr = 0
        do {
            ++it
            gerr = 0
            // update weights
            // hint: the dw is an array
            // that contains the values of
            // dcost(w0)/dw0, where w0 is
            // an array of the weights of 
            // the network...
            for (k = 0; k < examples.input.length; k++) {
                example = {input : examples.input[k], output : examples.output[k]}
                g = gradient(cost, w0)
                // Δw(l, i, j, t + 1) = 
                // Δw(l, i, j, t) 
                // * [g(l, i, j, t) / (g(l, i, j, t - 1) - g(l, i, j, t)]
                // g = J'(w)
                // dw(t + 1)
                dw = computedw(dw, g, gm1, mju, ita)
                console.log(dw)
                w0 = w0.map((wi, i) => {
                    return wi += dw[i]
                })
                // gm1 = g
                gm1.forEach((gi, i) => {
                    gi = g[i]
                })
                gerr += err(example)
            }
        } while (it < maxiterations && gerr > minerr)
        this.otherParameters['training error'] = err({input : examples.input[0], output : examples.output[0]})
        this.otherParameters.epochs = it
        return this
    }
    computeError(output) {
        let err = 0
        this.otherParameters
            .Layers[this.hiddenLayersSize.length + 1]
            .neurons
            .forEach((neuron, i) => {
                neuron.target = output[i]
                neuron.error = neuron.target - neuron.output
                err += (neuron.error * neuron.error)
            })
        this.otherParameters
            .Layers[this.hiddenLayersSize.length + 1]
            .squareError = err
        return err
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
        'method': 'backpropagation',
        'learning rate': 0.09,
        'momentum': 0.0,
        'max iterations': 10e5,
        'minimal error': 1e-4,
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
function setNetworkWeights(w, network) {
    // L number of layers, 
    //nl neurons in every layer
    let L = network
        .hiddenLayersSize
        .length + 2, l, k,
        neuronl, neuronlm1,
        q = 0, value,
        n = (l) => {
            return network
                .otherParameters
                .Layers[l].neurons.length
        }
    for (l = 1; l < L; l++) {
        for (neuronl = 0; neuronl < n(l); neuronl++) {
            for (neuronlm1 = -1; neuronlm1 < n(l - 1); neuronlm1++) {
                k = q + neuronl * (n(l - 1) + 1) + neuronlm1 + 1
                value = w[k]
                if (neuronlm1 === -1) network._setBias({
                    layer: l, neuron: neuronl, value
                })
                else network._setWeight({
                    source: { layer: l - 1, neuron: neuronlm1 },
                    target: { layer: l, neuron: neuronl },
                    value
                })
            }
        }
        q += n(l) * (n(l - 1) + 1)
    }
}
function orderWeightsTo1DimArray(network) {
    let L = network
        .hiddenLayersSize
        .length + 2, l, w = [],
        neuronl, neuronlm1,
        source, target,
        n = (l) => {
            return network
                .otherParameters
                .Layers[l].neurons.length
        }
    for (l = 1; l < L; l++) {
        for (neuronl = 0; neuronl < n(l); neuronl++) {
            for (neuronlm1 = -1; neuronlm1 < n(l - 1); neuronlm1++) {
                source = { layer: l - 1, neuron: neuronlm1 }
                target = { layer: l, neuron: neuronl }
                if (neuronlm1 === -1) w.push(network._getBias({ layer: l, neuron: neuronl }))
                else w.push(network._getWeight({ source, target }))
            }
        }
    }
    return w
}
function dataDoNotExists(network) {
    let dt = typeof network.otherParameters.data === 'undefined'
        || network.otherParameters.data === null
        || !(network.otherParameters.data.input instanceof Array
            && network.otherParameters.data.output instanceof Array)
    return dt
}
function areInsertedExampelesInOptions(opt) {
    let ex = false
    if (opt.examples instanceof Array) {
        if (opt.examples.every(example => {
            return typeof example.input !== 'undefined'
                && example.output !== 'undefined'
        })) ex = true
    }
    return ex
}
// Δw(l, i, j, t + 1) = 
// Δw(l, i, j, t) 
// * [g(l, i, j, t) / (g(l, i, j, t - 1) - g(l, i, j, t)]
// g = J'(w)
function computedw(dw, g, gm1, mju, ita) {
    let k, alpha
    for (k = 0; k < dw.length; k++) {
        if (dw[k] === 0.0) alpha = ita * g[k]
        alpha = g[k] / (gm1[k] - g[k])
        // if dw(t) and dw(t + 1) are
        // in the same direction...
        if (dw[k] * alpha > 0) {
            // if dw(t + 1) / dw(t) is
            // greater than the maximum
            // growth factor ...
            if (alpha > mju) dw[k] *= mju
            else dw[k] *= alpha
        }
    }
    return dw
}
function random(m, seed) {
    // use this to create controled
    // random numbers 
    var i, j, k, rn
    for (i = 0; i < m; i++) {
        seed = parseInt(seed);
        k = parseInt(seed / 127773);
        seed = parseInt(16807 * (seed - k * 127773) - k * 2836);
        if (seed < 0) seed += 2147483647;
        rn = seed * 4.656612875e-10;
    }
    // rn 
    return 2 * Math.random() - 1;
}
function sumarize(a, b) {
    return a + b
}
module.exports = { MLP }
