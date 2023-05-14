import * as tf from "@tensorflow/tfjs-node";
import dog1 from "./dog1.json";
import dog2 from "./dog2.json";
import cat1 from "./cat1.json";
const dog1Tensor = tf.tensor(dog1);
const dog2Tensor = tf.tensor(dog2);
const cat1Tensor = tf.tensor(cat1);
const compareCosineSimilarity = (tensor1, tensor2) => {
    const dotProduct = tensor1.dot(tensor2);
    const norm1 = tensor1.norm();
    const norm2 = tensor2.norm();
    const cosineSimilarity = dotProduct.div(norm1.mul(norm2));
    return cosineSimilarity.dataSync()[0];
};
console.log("dog1 vs dog2", compareCosineSimilarity(dog1Tensor, dog2Tensor));
console.log("dog1 vs cat1", compareCosineSimilarity(dog1Tensor, cat1Tensor));
