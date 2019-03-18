'use strict'
let {MLP} = require('MLP')
let nnet = {
   MLP
   // recurrent
   // SOM
   // APT
   // RBF
   // SVM
   // GHA, APEX
}
let net = new nnet.MLP({architecture : '2-2-1'})
   .data({input : [ [0, 0], [1, 0],[1,1], [0, 1] ], output : [[0], [1], [0], [1]]})
   .train({"method" : "backpropagation"})
net.activate([1, 0]).computeError([1])
//net.activate([1, 1]).computeError([0])
console.log(net)
