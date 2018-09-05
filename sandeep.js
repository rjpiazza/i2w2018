console.log('sandeep.js is loaded');

const model = tf.sequential();

model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
}));

model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
});
const xs = tf.tensor([1, 3, 5, 7], [4, 1]);
const ys = tf.tensor([2, 4, 6, 8], [4, 1]);

model.fit(xs, ys).then(()=>{
    model.predict(tf.tensor([1, 2, 3, 4], [4, 1])).print();
}
);
